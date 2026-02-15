import dimod
from collections import defaultdict

def square_bqm_to_binary_polynomial(bqm: dimod.BinaryQuadraticModel):
    """
    Square a dimod BinaryQuadraticModel and return a dimod.BinaryPolynomial.
    Higher-order terms are preserved.
    """

    if bqm.vartype is not dimod.BINARY:
        raise ValueError("BQM must have vartype=BINARY")

    # Collect polynomial terms: monomial -> coefficient
    poly = defaultdict(float)

    # constant term
    if bqm.offset != 0.0:
        poly[()] += bqm.offset

    # linear terms
    for v, bias in bqm.linear.items():
        poly[(v,)] += bias

    # quadratic terms
    for (u, v), bias in bqm.quadratic.items():
        poly[tuple(sorted((u, v)))] += bias

    # ---- square the polynomial ----
    result = defaultdict(float)
    terms = list(poly.items())
    n = len(terms)

    def multiply_terms(t1, t2):
        # binary variables: x^2 = x
        return tuple(sorted(set(t1) | set(t2)))

    # diagonal terms
    for term, coeff in terms:
        result[term] += coeff * coeff

    # cross terms (2ab)
    for i in range(n):
        t1, c1 = terms[i]
        for j in range(i + 1, n):
            t2, c2 = terms[j]
            new_term = multiply_terms(t1, t2)
            result[new_term] += 2 * c1 * c2

    # ---- build BinaryPolynomial ----
    # NOTE: constant term is stored under key ()
    bp = dimod.BinaryPolynomial(dict(result), vartype=dimod.BINARY)

    return bp
