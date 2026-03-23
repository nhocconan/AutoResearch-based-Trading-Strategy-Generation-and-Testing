#!/usr/bin/env python3
"""
Experiment #694: 4h Primary + 1d HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: After 607 failed strategies, the pattern shows:
1. RSI/CRSI mean-reversion works but gets whipsawed in strong trends
2. Fisher Transform is specifically designed for reversal detection in bear/range markets
3. Fisher crosses at extremes (-1.5, +1.5) have higher precision than RSI thresholds
4. Combined with Choppiness regime filter + 1d HMA trend = fewer false signals

This strategy uses:
- Ehlers Fisher Transform (period=9) for precise reversal entries
- Choppiness Index (14) to detect range vs trend regime
- 1d HMA (21) for major trend bias
- Asymmetric entries: Fisher reversals in chop, trend continuation when trending

Why this might beat Sharpe=0.520:
- Fisher Transform normalizes price to Gaussian distribution, better for extremes
- 4h timeframe = 25-45 trades/year (optimal per Rule 10)
- Choppiness filter prevents entering against strong trends
- 1d HMA keeps us on right side of major moves
- Conservative sizing (0.30) + ATR stop controls drawdown

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 25-45 trades/year on 4h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_hma_1d_v1"
timeframe = "4h"
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
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    Interpretation:
    - CHOP > 61.8: Range/consolidation (mean-revert)
    - CHOP < 38.2: Trending (trend-follow)
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    
    Steps:
    1. Calculate typical price: (high + low + close) / 3
    2. Normalize to -1 to +1 range using highest high / lowest low over period
    3. Apply Fisher transform: 0.5 * ln((1 + value) / (1 - value))
    4. Signal line = 1-period lag of Fisher
    
    Entry signals:
    - Long: Fisher crosses above -1.5 (oversold reversal)
    - Short: Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(close)
    
    # Typical price
    typical = (high + low + close) / 3.0
    typical_s = pd.Series(typical)
    
    # Normalize price to -1 to +1 range
    highest = typical_s.rolling(window=period, min_periods=period).max().values
    lowest = typical_s.rolling(window=period, min_periods=period).min().values
    
    price_range = highest - lowest
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = 2.0 * (typical - lowest) / (price_range + 1e-10) - 1.0
    
    # Clip to avoid log domain errors
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]) or np.isnan(hma_4h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (HMA slope over 5 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop_14[i] > 55.0  # Range/consolidation
        is_trend = chop_14[i] < 45.0  # Trending
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        # Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Fisher extreme levels (for range market entries)
        fisher_extreme_low = fisher[i] < -1.8
        fisher_extreme_high = fisher[i] > 1.8
        
        # === 4H HMA SLOPE (3 bars) ===
        hma_4h_slope_bull = hma_4h[i] > hma_4h[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h[i] < hma_4h[i-3] if i >= 3 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market (CHOP > 55) + Fisher extreme low = mean revert long
        if is_range and fisher_extreme_low:
            new_signal = POSITION_SIZE
        
        # Regime 2: Range market + Fisher cross above -1.5 = reversal long
        elif is_range and fisher_long_cross:
            new_signal = POSITION_SIZE
        
        # Regime 3: Trending market (CHOP < 45) + 1d bull + 4h bull + Fisher pullback
        elif is_trend and hma_1d_slope_bull and price_above_hma_1d:
            if hma_4h_slope_bull and fisher[i] < -0.5:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market (CHOP > 55) + Fisher extreme high = mean revert short
        if is_range and fisher_extreme_high:
            new_signal = -POSITION_SIZE
        
        # Regime 2: Range market + Fisher cross below +1.5 = reversal short
        elif is_range and fisher_short_cross:
            new_signal = -POSITION_SIZE
        
        # Regime 3: Trending market (CHOP < 45) + 1d bear + 4h bear + Fisher pullback
        elif is_trend and hma_1d_slope_bear and price_below_hma_1d:
            if hma_4h_slope_bear and fisher[i] > 0.5:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals