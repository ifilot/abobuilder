class ElementTable:
    def __init__(self):
        # Element symbol list indexed by atomic number
        self._elements = [
            None,  # atomic number 0 doesn't exist
            "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
            "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
            "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
            "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
            "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
            "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
            "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
            "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
            "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
            "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
            "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
            "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
        ]

        # Symbol to atomic number dictionary
        self._symbol_to_number = {symbol: i for i, symbol in enumerate(self._elements) if symbol}

    def atomic_number_from_element(self, symbol: str) -> int:
        if not isinstance(symbol, str):
            raise TypeError(f"Expected a string for the element symbol, got {type(symbol).__name__}")
        return self._symbol_to_number.get(symbol.capitalize(), None)

    def element_from_atomic_number(self, number: int) -> str:
        """Return element symbol given atomic number."""
        if 1 <= number < len(self._elements):
            return self._elements[number]
        return None
