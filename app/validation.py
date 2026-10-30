"""
Input Validation and Financial Health Ratio Engine for EMIPredict AI.

Enforces business logic rules, domain bounds, and consistency checks on applicant inputs.
Distinguishes between hard validation errors (which block model inference) and
soft underwriting warnings (which alert the user or loan officer).
"""

from typing import Dict, Any, List, Tuple


def validate_applicant_inputs(data: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
    """
    Validates applicant data dictionary.
    Returns: (is_valid, errors_list, warnings_list)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Age and Employment Consistency
    age = float(data.get("age", 0.0))
    if age < 18.0:
        errors.append(f"Applicant age ({int(age)}) must be at least 18 years.")
    elif age > 100.0:
        errors.append(f"Applicant age ({int(age)}) exceeds reasonable upper limit of 100 years.")

    years_emp = float(data.get("years_of_employment", 0.0))
    if years_emp < 0.0:
        errors.append("Years of employment cannot be negative.")
    elif age >= 18.0 and years_emp > (age - 18.0):
        errors.append(
            f"Years of employment ({years_emp:.1f}) cannot exceed implied working years "
            f"since age 18 ({age - 18.0:.1f} years)."
        )

    # 2. Income and Financial Amounts
    salary = float(data.get("monthly_salary", 0.0))
    if salary <= 0.0:
        errors.append("Monthly salary must be strictly greater than ₹0.")
    elif salary < 5000.0:
        warnings.append("Reported monthly salary is below ₹5,000, which falls below statutory minimum wage in most Indian states.")

    req_amount = float(data.get("requested_amount", 0.0))
    if req_amount <= 0.0:
        errors.append("Requested loan amount must be strictly greater than ₹0.")

    req_tenure = float(data.get("requested_tenure", 0.0))
    if req_tenure <= 0.0:
        errors.append("Requested loan tenure must be strictly greater than 0 months.")
    elif req_tenure < 3.0 or req_tenure > 120.0:
        warnings.append(f"Requested tenure of {int(req_tenure)} months is outside standard retail loan tenures (3 to 84 months).")

    # 3. Credit Score
    credit_score = float(data.get("credit_score", 0.0))
    if credit_score < 0.0:
        errors.append("Credit score cannot be negative.")
    elif credit_score > 900.0:
        warnings.append(
            f"Credit score of {int(credit_score)} exceeds standard Indian bureau scale (300–900). "
            "Scores up to 1,200 were observed in the synthetic generator and are accepted without clipping."
        )
    elif credit_score == 0.0:
        warnings.append("Credit score is 0, indicating a New-To-Credit (NTC) applicant with no formal bureau history.")

    # 4. Existing Loans vs Current EMI Consistency
    existing_loans = str(data.get("existing_loans", "No")).strip().capitalize()
    current_emi = float(data.get("current_emi_amount", 0.0))

    if current_emi < 0.0:
        errors.append("Current monthly EMI amount cannot be negative.")

    if existing_loans == "No" and current_emi > 0.0:
        warnings.append(
            f"Existing loans is marked 'No' but Current EMI is ₹{current_emi:,.2f}. "
            "For consistency with model training data, Current EMI will be treated as ₹0.00."
        )
    elif existing_loans == "Yes" and current_emi == 0.0:
        warnings.append("Existing loans is marked 'Yes' but Current EMI is ₹0. Please confirm if existing loans are active.")

    # 5. House Type vs Rent Consistency
    house_type = str(data.get("house_type", "Rented")).strip()
    rent = float(data.get("monthly_rent", 0.0))
    if rent < 0.0:
        errors.append("Monthly rent cannot be negative.")
    elif house_type in ["Own", "Family"] and rent > 0.0:
        warnings.append(f"Property is '{house_type}' but monthly rent of ₹{rent:,.2f} was reported.")

    # 6. Expenses non-negativity
    expense_fields = [
        ("school_fees", "School fees"),
        ("college_fees", "College fees"),
        ("travel_expenses", "Travel expenses"),
        ("groceries_utilities", "Groceries & utilities"),
        ("other_monthly_expenses", "Other monthly expenses"),
        ("bank_balance", "Bank balance"),
        ("emergency_fund", "Emergency fund"),
        ("dependents", "Dependents count")
    ]
    for field_key, field_name in expense_fields:
        val = float(data.get(field_key, 0.0))
        if val < 0.0:
            errors.append(f"{field_name} cannot be negative.")

    # 7. Liquid Reserves Check
    bank_bal = float(data.get("bank_balance", 0.0))
    emerg_fund = float(data.get("emergency_fund", 0.0))
    if bank_bal < emerg_fund:
        warnings.append(
            f"Liquid bank balance (₹{bank_bal:,.2f}) is lower than reported emergency fund (₹{emerg_fund:,.2f})."
        )

    # 8. High Leverage / FOIR Warning
    if salary > 0.0 and req_tenure > 0.0:
        implied_inst = req_amount / req_tenure
        effective_emi = current_emi if existing_loans == "Yes" else 0.0
        projected_foir = ((effective_emi + implied_inst) / salary) * 100.0
        if projected_foir > 65.0:
            warnings.append(
                f"Projected Debt-to-Income (FOIR) is {projected_foir:.1f}%. "
                "Values above 50% indicate severe debt burden under standard Indian banking guidelines."
            )

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def calculate_financial_ratios(data: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes standard retail banking underwriting metrics and ratios.
    """
    salary = float(data.get("monthly_salary", 0.0))
    rent = float(data.get("monthly_rent", 0.0))
    house_type = str(data.get("house_type", "Rented")).strip()
    if house_type in ["Own", "Family"] and rent > 0.0:
        # Keep reported rent or treat as other housing cost
        pass

    school_fees = float(data.get("school_fees", 0.0))
    college_fees = float(data.get("college_fees", 0.0))
    travel = float(data.get("travel_expenses", 0.0))
    groceries = float(data.get("groceries_utilities", 0.0))
    other_exp = float(data.get("other_monthly_expenses", 0.0))

    existing_loans = str(data.get("existing_loans", "No")).strip().capitalize()
    raw_emi = float(data.get("current_emi_amount", 0.0))
    current_emi = raw_emi if existing_loans == "Yes" else 0.0

    req_amount = float(data.get("requested_amount", 0.0))
    req_tenure = max(float(data.get("requested_tenure", 1.0)), 1.0)

    # Computations
    total_living_expenses = rent + school_fees + college_fees + travel + groceries + other_exp
    total_obligations = total_living_expenses + current_emi
    disposable_income = salary - total_obligations

    implied_installment = req_amount / req_tenure

    current_foir = (current_emi / salary * 100.0) if salary > 0 else 0.0
    projected_foir = ((current_emi + implied_installment) / salary * 100.0) if salary > 0 else 0.0

    # Savings-to-salary ratio
    bank_balance = float(data.get("bank_balance", 0.0))
    emergency_fund = float(data.get("emergency_fund", 0.0))
    liquidity_months = (bank_balance + emergency_fund) / total_obligations if total_obligations > 0 else 0.0

    return {
        "monthly_salary": round(salary, 2),
        "total_living_expenses": round(total_living_expenses, 2),
        "total_obligations": round(total_obligations, 2),
        "disposable_income": round(disposable_income, 2),
        "implied_installment": round(implied_installment, 2),
        "current_foir": round(current_foir, 2),
        "projected_foir": round(projected_foir, 2),
        "liquidity_buffer_months": round(liquidity_months, 1)
    }
