use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Token(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CompactPatternToken {
    token: u32,
}

impl CompactPatternToken {
    pub const fn new_token(token: Token) -> Self {
        CompactPatternToken { token: token.0 }
    }

    pub const fn new_ignore() -> Self {
        CompactPatternToken { token: u32::MAX }
    }

    pub const fn matches(self, other: Token) -> bool {
        self.token == other.0 || self.token == u32::MAX
    }

    pub const fn regular(self) -> Option<Token> {
        if self.token == u32::MAX {
            None
        } else {
            Some(Token(self.token))
        }
    }

    pub const fn unpack(self) -> PatternToken {
        if self.token == u32::MAX {
            PatternToken::Ignore
        } else {
            PatternToken::Regular(Token(self.token))
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PatternToken {
    Regular(Token),
    Ignore, // Ignore should be last so the ordering matches the CompactToken ordering
}

impl PatternToken {
    pub const fn compact(self) -> CompactPatternToken {
        match self {
            PatternToken::Ignore => CompactPatternToken::new_ignore(),
            PatternToken::Regular(token) => CompactPatternToken::new_token(token),
        }
    }

    pub const fn is_regular(self) -> bool {
        match self {
            PatternToken::Regular(_) => true,
            PatternToken::Ignore => false,
        }
    }

    pub const fn is_ignore(self) -> bool {
        !self.is_regular()
    }
}

impl PartialEq<CompactPatternToken> for PatternToken {
    fn eq(&self, other: &CompactPatternToken) -> bool {
        match self {
            PatternToken::Ignore => other.token == u32::MAX,
            PatternToken::Regular(token) => token.0 == other.token,
        }
    }
}

impl PartialEq<PatternToken> for CompactPatternToken {
    fn eq(&self, other: &PatternToken) -> bool {
        other == self
    }
}
