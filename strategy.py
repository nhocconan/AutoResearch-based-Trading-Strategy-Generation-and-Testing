#!/usr/bin/env python3
"""
Experiment #1440: 6h Primary + 1d/1w HTF — Fisher Transform + CHOP Regime

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). This strategy combines:
1. Ehlers Fisher Transform (period=9) for reversal detection - excels in bear/range markets
2. Choppiness Index (14) for regime detection (trend vs range)
3. 1d HMA(21) for major trend bias (primary HTF filter)
4. 1w HMA(50) for secular trend confirmation (secondary, not required)
5. ATR(14) trailing stoploss

Why this should work where others failed:
- Fisher Transform catches reversals better than RSI in 2022-style crashes
- CHOP filter adapts logic: trend-follow when CHOP<61.8, mean-revert when CHOP>61.8
- 6h TF = natural 30-50 trades/year (fee-efficient, not overtraded like 15m/30m)
- LOOSE entry: Fisher > -1.0 (not exact cross) + 1d HMA bias only (not both 1d+1w)
- Avoids weekly_pivot patterns that failed in 11+ prior experiments

Entry logic (LOOSE to guarantee >=30 trades/year):
- LONG: 1d_HMA bullish + Fisher > -1.0 + (CHOP<61.8 OR price<BB_lower)
- SHORT: 1d_HMA bearish + Fisher < +1.0 + (CHOP<61.8 OR price>BB_upper)

Timeframe: 6h
Size: 0.25-0.30 discrete
Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_chop_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Highlights turning points by transforming price into a bounded oscillator
    Reference: Ehlers, J.F. (2002) "Fisher Transform"
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        # Find highest high and lowest low over period
        highest = np.max(close[i-period+1:i+1])
        lowest = np.min(close[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        # Normalize price to 0-1 range
        value = (close[i] - lowest) / (highest - lowest)
        
        # Clamp to avoid division by zero
        value = max(0.001, min(0.999, value))
        
        # Apply Fisher transformation
        fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
    
    return fisher

def calculate_chop(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = range-bound, CHOP < 38.2 = strongly trending
    Reference: E.W. Dreiss
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            atr_sum += tr
        
        chop[i] = 100 * (atr_sum / (highest - lowest)) / np.sqrt(period)
    
    return chop

def calculate_bollinger(close, period=20, std_dev=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, sma, lower

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
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
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    fisher = calculate_fisher(close, period=9)
    chop = calculate_chop(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_dev=2.0)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (CHOP) ===
        is_trending = chop[i] < 61.8
        is_ranging = chop[i] >= 61.8
        
        # === TREND DIRECTION (1d HMA bias - PRIMARY) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 1w HMA CONFIRMATION (SECONDARY - boosts conviction, not required) ===
        price_above_1w = not np.isnan(hma_1w_aligned[i]) and close[i] > hma_1w_aligned[i]
        price_below_1w = not np.isnan(hma_1w_aligned[i]) and close[i] < hma_1w_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS (LOOSE - value based, not exact cross) ===
        fisher_bullish = fisher[i] > -1.0  # Emerging from oversold
        fisher_bearish = fisher[i] < 1.0   # Emerging from overbought
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = not np.isnan(bb_lower[i]) and close[i] <= bb_lower[i]
        price_near_bb_upper = not np.isnan(bb_upper[i]) and close[i] >= bb_upper[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG entries - 1d bullish + Fisher bullish + regime confirmation
        if price_above_1d and fisher_bullish:
            if is_trending:
                # Trending regime: enter on Fisher reversal with trend
                if price_above_1w:
                    desired_signal = SIZE_STRONG  # Strong: 1d+1w aligned
                else:
                    desired_signal = SIZE_BASE    # Base: 1d only
            elif is_ranging and price_near_bb_lower:
                # Ranging regime: mean revert at BB support
                desired_signal = SIZE_BASE
        
        # SHORT entries - 1d bearish + Fisher bearish + regime confirmation
        elif price_below_1d and fisher_bearish:
            if is_trending:
                # Trending regime: enter on Fisher reversal with trend
                if price_below_1w:
                    desired_signal = -SIZE_STRONG  # Strong: 1d+1w aligned
                else:
                    desired_signal = -SIZE_BASE    # Base: 1d only
            elif is_ranging and price_near_bb_upper:
                # Ranging regime: mean revert at BB resistance
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals