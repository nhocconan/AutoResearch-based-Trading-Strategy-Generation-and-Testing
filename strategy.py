#!/usr/bin/env python3
"""
Experiment #010: 1d Williams %R + 1w SMA + Choppiness Regime

Hypothesis: Williams %R is more responsive than RSI at catching reversals.
Very tight thresholds (extreme only) + 1w trend bias + chop regime filter.
Target: 30-60 total trades over 4 years (avoid overtrading failure mode).

Why it should work in BOTH bull AND bear:
- Bull market: Price > 1w SMA + Williams %R touches -80 (oversold bounce)
- Bear market: Price < 1w SMA + Williams %R touches -20 (failed rally = short)
- Range market: Choppiness filter prevents trend-chasing entries
- Williams %R 0 to -20 = overbought, -80 to -100 = oversold (inverted scale)

Key design choices to avoid overtrading:
1. ONLY enter on Williams %R extreme touches (not crosses)
2. Require 1w SMA trend alignment (reduces false entries)
3. Require Choppiness confirmation (chop < 38.2 for trending, BB squeeze for ranging)
4. Discrete sizes: 0.25 (normal), 0.30 (strong confluence)

Target: Sharpe > 0.5, trades 30-60 train, trades >= 10 test
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_williams_r_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_williams_r(high, low, close, period=20):
    """
    Williams %R - momentum indicator measuring overbought/oversold
    Scale: 0 to -100 (inverted from typical oscillators)
    -0 to -20 = overbought
    -80 to -100 = oversold
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    williams = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        range_val = highest_high - lowest_low
        
        if range_val < 1e-10:
            williams[i] = williams[i-1] if i > 0 else -50
            continue
        
        williams[i] = -100.0 * (highest_high - close[i]) / range_val
    
    return williams

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w SMA(50) for trend bias - aligned to 1d
    sma_1w_raw = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_raw)
    
    # Calculate 1d indicators
    williams_r = calculate_williams_r(high, low, close, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
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
    
    # Warmup period for 1w SMA(50) = ~1 year of 1d data
    min_bars = 300
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(williams_r[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1w SMA bias) ===
        price_above_1w = close[i] > sma_1w_aligned[i]
        price_below_1w = close[i] < sma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === WILLIAMS %R SIGNALS (tight thresholds) ===
        williams = williams_r[i]
        
        # Williams %R extremes (NOT crosses - extremes are more reliable)
        # -80 to -100 = oversold, -0 to -20 = overbought
        williams_oversold = williams <= -80  # Strong oversold
        williams_overbought = williams >= -20  # Strong overbought
        
        # === BOLLINGER BAND TOUCH (additional confluence) ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.01
        bb_touch_upper = close[i] >= bb_upper[i] * 0.99
        
        # === ENTRY LOGIC (VERY TIGHT - prevent overtrading) ===
        desired_signal = 0.0
        confluence_count = 0
        
        # TREND REGIME: Only enter in direction of 1w trend
        if is_trend_regime:
            # LONG: 1w bullish + Williams oversold
            if price_above_1w and williams_oversold:
                confluence_count = 1
                if bb_touch_lower:
                    confluence_count = 2
                desired_signal = SIZE_STRONG if confluence_count >= 2 else SIZE_BASE
            
            # SHORT: 1w bearish + Williams overbought
            elif price_below_1w and williams_overbought:
                confluence_count = 1
                if bb_touch_upper:
                    confluence_count = 2
                desired_signal = -SIZE_STRONG if confluence_count >= 2 else -SIZE_BASE
        
        # RANGE REGIME: Mean reversion from extremes
        elif is_range_regime:
            # LONG: Williams oversold + BB touch (bounce from lower band)
            if williams_oversold and bb_touch_lower:
                desired_signal = SIZE_BASE
            
            # SHORT: Williams overbought + BB touch (fall from upper band)
            elif williams_overbought and bb_touch_upper:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Very conservative - only strong signals
        else:
            # LONG: Strong oversold + BB touch + above 1w SMA
            if williams_oversold and bb_touch_lower and price_above_1w:
                desired_signal = SIZE_BASE
            
            # SHORT: Strong overbought + BB touch + below 1w SMA
            elif williams_overbought and bb_touch_upper and price_below_1w:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (3x ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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