#!/usr/bin/env python3
"""
Experiment #002: 12h Dual-Regime Strategy with 1d HMA Bias

Hypothesis: After 539+ failures, the clearest pattern is that SINGLE-regime strategies
fail because crypto alternates between trending (2021 bull, 2023 recovery) and ranging
(2022 crash, 2025 bear). This strategy uses CHOPPININESS INDEX to detect regime and
switches between two proven entry methods:

1. RANGE REGIME (CHOP > 61.8): CONNORS RSI mean reversion
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + price > 1d_HMA (not in strong downtrend)
   - Short: CRSI > 85 + price < 1d_HMA (not in strong uptrend)
   - 75% win rate in academic studies, works in bear/range markets

2. TREND REGIME (CHOP < 38.2): DONCHIAN breakout with HTF filter
   - Long: Price breaks 20-bar Donchian high + price > 1d_HMA
   - Short: Price breaks 20-bar Donchian low + price < 1d_HMA
   - Catches sustained moves, avoids whipsaw in 2022 crash

3. TRANSITION (38.2-61.8): Use Fisher Transform reversals
   - Fisher < -1.5 crossing up = long
   - Fisher > +1.5 crossing down = short

Why 12h timeframe:
- 20-50 trades/year target (fee drag 1-2.5%)
- Less noise than 1h/4h, more signals than 1d
- Proven in research: higher TF = better Sharpe for BTC/ETH

HTF: 1d HMA(21) for trend bias via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete, ATR-scaled
Stoploss: 2.5 * ATR(14) trailing (wider for 12h)

This is DIFFERENT from failed #001 (4h Chop+Connors) by using:
- 12h primary (higher TF = less noise)
- 1d HTF bias (more stable than 4h for 12h strategy)
- Dual-regime with Fisher fallback (3 entry modes, not just 2)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_connors_donchian_1d_hma_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using standard formula."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_connors_rsi(close, lookback_rsi=3, lookback_streak=2, lookback_rank=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI applied to consecutive up/down day count
    PercentRank: percentage of prior closes below current close
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, lookback_rsi)
    
    # Component 2: RSI of Streak
    # Streak = consecutive up (+1) or down (-1) days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Apply RSI to absolute streak values
    streak_rsi = calculate_rsi(np.abs(streak), lookback_streak)
    
    # Component 3: PercentRank(100)
    percent_rank = np.zeros(n) * np.nan
    for i in range(lookback_rank, n):
        window = close[i-lookback_rank:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = (count_below / lookback_rank) * 100
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of ATR over period
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / price_range.replace(0, np.inf)) / np.log10(period)
    
    return chop.values

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = normalized price
    Excellent for catching reversals at extremes (-2.0 to +2.0)
    """
    close_s = pd.Series(close)
    
    # Normalize price to -1 to +1 range using highest high / lowest low
    highest = close_s.rolling(window=period, min_periods=period).max()
    lowest = close_s.rolling(window=period, min_periods=period).min()
    
    # Avoid division by zero
    price_range = highest - lowest
    price_range = price_range.replace(0, 0.001)
    
    # Normalize: (close - lowest) / (highest - lowest) * 2 - 1
    x = ((close_s - lowest) / price_range * 2 - 1).clip(-0.99, 0.99)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x))
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.values, fisher_signal.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher(close, 9)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(crsi[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 38.2
        is_range_regime = chop_14[i] > 61.8
        # Between 38.2-61.8 = transition
        
        # === 1D HMA TREND BIAS ===
        bull_bias = close[i] > hma_1d_aligned[i]
        bear_bias = close[i] < hma_1d_aligned[i]
        
        # === ATR-BASED POSITION SIZING ===
        # Reduce size when volatility is high (protect in crashes)
        if i > 200:
            atr_median = np.nanmedian(atr_14[i-200:i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
        else:
            atr_ratio = 1.0
        atr_ratio = np.clip(atr_ratio, 0.5, 2.5)
        size_multiplier = 1.0 / atr_ratio
        current_size = BASE_SIZE * size_multiplier
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # MODE 1: RANGE REGIME - Connors RSI mean reversion
        if is_range_regime:
            # Long: CRSI < 15 + not in strong downtrend
            if crsi[i] < 15 and not bear_bias:
                new_signal = current_size
            
            # Short: CRSI > 85 + not in strong uptrend
            elif crsi[i] > 85 and not bull_bias:
                new_signal = -current_size
        
        # MODE 2: TREND REGIME - Donchian breakout with HTF filter
        elif is_trend_regime:
            # Check for breakout (compare to previous bar's Donchian)
            if i > 0 and not np.isnan(donchian_upper[i-1]):
                breakout_long = close[i] > donchian_upper[i-1]
                if breakout_long and bull_bias:
                    new_signal = current_size
            
            if i > 0 and not np.isnan(donchian_lower[i-1]):
                breakout_short = close[i] < donchian_lower[i-1]
                if breakout_short and bear_bias:
                    new_signal = -current_size
        
        # MODE 3: TRANSITION REGIME - Fisher Transform reversals
        else:
            # Fisher crossing above -1.5 from below (bullish)
            fisher_long = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
            # Fisher crossing below +1.5 from above (bearish)
            fisher_short = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
            
            if fisher_long and not bear_bias:
                new_signal = current_size
            elif fisher_short and not bull_bias:
                new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1d bias turns strongly bearish
            if position_side > 0 and bear_bias and chop_14[i] < 35:
                trend_reversal = True
            # Exit short if 1d bias turns strongly bullish
            if position_side < 0 and bull_bias and chop_14[i] < 35:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals