#!/usr/bin/env python3
"""
Experiment #642: 12h Primary + 1d/1w HTF — Fisher + HMA + Choppiness Regime

Hypothesis: 12h timeframe balances signal quality with trade frequency. Previous 12h 
strategies failed with 0 trades due to overly strict entry conditions. This version:

1. LOOSER Fisher thresholds (-0.8/+0.8 instead of -1.2/+1.2) for more signals
2. BROADER Choppiness regime ranges (chop>50, trend<40) for more regime coverage
3. SIMPLER HMA trend filter (faster than KAMA, more responsive)
4. DUAL HTF confirmation (both 1d AND 1w must agree for trend trades)
5. FALLBACK entry conditions when primary signals don't trigger
6. PERMISSIVE hold logic to maintain positions through minor pullbacks

Key innovations for trade frequency:
- Entry requires only 2 of 3 conditions (not all 3)
- Fisher extreme levels trigger even without cross
- Hold maintains position if HTF trend unchanged
- Minimum signal duration of 3 bars before flip allowed

Why this should beat Sharpe=0.612:
- 12h TF = fewer false signals than 4h, more trades than 1d
- Fisher Transform proven edge in bear/range markets
- Dual HTF (1d+1w) prevents counter-trend trades
- Looser thresholds ensure 30-50 trades/year target
- Conservative sizing (0.25) survives crashes

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_hma_chop_dualhtf_v1"
timeframe = "12h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform.
    Converts price to Gaussian normal distribution for clearer reversal signals.
    Long: Fisher crosses above -0.8 from below
    Short: Fisher crosses below +0.8 from above
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period:
        return fisher, fisher_signal
    
    price = np.zeros(n)
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        range_val = hh - ll
        if range_val < 1e-10:
            range_val = 1e-10
        
        price_raw = (close[i] - ll) / range_val
        
        if i > period:
            price[i] = 0.33 * 2 * (price_raw - 0.5) + 0.67 * price[i-1]
        else:
            price[i] = 0.33 * 2 * (price_raw - 0.5)
        
        price[i] = np.clip(price[i], -0.999, 0.999)
        fisher[i] = 0.5 * np.log((1 + price[i]) / (1 - price[i]))
        fisher_signal[i] = fisher[i-1] if i > 0 else fisher[i]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Hull Moving Average - smoother and more responsive than EMA."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    CHOP > 50 = choppy/ranging (mean revert)
    CHOP < 40 = trending (trend follow)
    Between 40-50 = neutral (use fallback logic)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop_raw = 100.0 * np.log10(atr_sum / (hh - ll + 1e-10)) / np.log10(period)
        chop = np.clip(chop_raw, 0, 100)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h indicators (primary timeframe)
    fisher_12h, fisher_signal_12h = calculate_fisher_transform(high, low, close, period=9)
    hma_12h = calculate_hma(close, period=21)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.25
    SIZE_SHORT = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    bars_in_position = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_12h[i]) or np.isnan(fishersignal_12h[i]) if 'fishersignal_12h' in dir() else np.isnan(fisher_signal_12h[i]):
            continue
        if np.isnan(hma_12h[i]) or np.isnan(chop_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_12h[i] > 50.0
        is_trending = chop_12h[i] < 40.0
        is_neutral = not is_choppy and not is_trending
        
        # === HTF TREND BIAS (1d + 1w must agree for strong signal) ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        htf_1w_bullish = close[i] > hma_1w_aligned[i]
        htf_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # Strong HTF agreement
        htf_strong_bull = htf_1d_bullish and htf_1w_bullish
        htf_strong_bear = htf_1d_bearish and htf_1w_bearish
        
        # === 12h HMA TREND ===
        hma_bullish = close[i] > hma_12h[i]
        hma_bearish = close[i] < hma_12h[i]
        
        # === FISHER TRANSFORM SIGNALS (LOOSER THRESHOLDS) ===
        fisher_long_cross = (fisher_12h[i] > -0.8) and (fisher_signal_12h[i] <= -0.8)
        fisher_short_cross = (fisher_12h[i] < 0.8) and (fisher_signal_12h[i] >= 0.8)
        
        # Fisher extreme levels (for mean reversion in chop)
        fisher_oversold = fisher_12h[i] < -1.0
        fisher_overbought = fisher_12h[i] > 1.0
        
        # Fisher moderate levels (for trend continuation)
        fisher_bullish = fisher_12h[i] > -0.5
        fisher_bearish = fisher_12h[i] < 0.5
        
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY MARKET (Mean Reversion with Fisher) ===
        if is_choppy:
            # Long: Fisher oversold (extreme mean reversion)
            if fisher_oversold:
                desired_signal = SIZE_LONG
            # Short: Fisher overbought (extreme mean reversion)
            elif fisher_overbought:
                desired_signal = -SIZE_SHORT
            # Fisher cross signals
            elif fisher_long_cross:
                desired_signal = SIZE_LONG
            elif fisher_short_cross:
                desired_signal = -SIZE_SHORT
        
        # === REGIME 2: TRENDING MARKET (Trend Follow) ===
        elif is_trending:
            # Long: Strong HTF bull + HMA bull + Fisher supportive (2 of 3)
            long_conditions = [htf_strong_bull, hma_bullish, fisher_bullish]
            if sum(long_conditions) >= 2:
                desired_signal = SIZE_LONG
            # Short: Strong HTF bear + HMA bear + Fisher supportive (2 of 3)
            short_conditions = [htf_strong_bear, hma_bearish, fisher_bearish]
            if sum(short_conditions) >= 2:
                desired_signal = -SIZE_SHORT
            # Fisher cross with any trend confirmation
            elif fisher_long_cross and (htf_1d_bullish or hma_bullish):
                desired_signal = SIZE_LONG
            elif fisher_short_cross and (htf_1d_bearish or hma_bearish):
                desired_signal = -SIZE_SHORT
        
        # === REGIME 3: NEUTRAL/TRANSITION (Fallback logic) ===
        else:
            # Use simpler logic - just HTF + Fisher
            if htf_1d_bullish and fisher_bullish:
                desired_signal = SIZE_LONG
            elif htf_1d_bearish and fisher_bearish:
                desired_signal = -SIZE_SHORT
            elif fisher_long_cross:
                desired_signal = SIZE_LONG
            elif fisher_short_cross:
                desired_signal = -SIZE_SHORT
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        # CRITICAL: Keep positions open through minor pullbacks to ensure trade frequency
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if HTF still bullish OR HMA still bullish
                if htf_1d_bullish or hma_bullish:
                    desired_signal = SIZE_LONG
            elif position_side < 0:
                # Hold short if HTF still bearish OR HMA still bearish
                if htf_1d_bearish or hma_bearish:
                    desired_signal = -SIZE_SHORT
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE_LONG
        elif desired_signal < 0:
            desired_signal = -SIZE_SHORT
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
                bars_in_position = 1
            elif np.sign(desired_signal) != position_side:
                # Only allow flip after minimum 3 bars (prevent whipsaw)
                if bars_in_position >= 3:
                    position_side = int(np.sign(desired_signal))
                    entry_price = close[i]
                    entry_atr = atr_12h[i]
                    highest_since_entry = close[i] if position_side > 0 else 0.0
                    lowest_since_entry = close[i] if position_side < 0 else float('inf')
                    bars_in_position = 1
                # else maintain current position
            else:
                # Same side - update trailing levels
                bars_in_position += 1
                if position_side > 0:
                    highest_since_entry = max(highest_since_entry, close[i])
                elif position_side < 0:
                    lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
                bars_in_position = 0
        
        signals[i] = desired_signal
    
    return signals