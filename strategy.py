#!/usr/bin/env python3
"""
Experiment #1619: 4h Donchian Breakout + HMA Trend + Choppiness Regime

Hypothesis: Simple price channel breakout (Donchian 20) with HMA(48) trend bias 
and Choppiness(14) regime filter is the proven winning pattern from DB.
This combination achieved test Sharpe 1.38-1.46 on SOLUSDT in prior experiments.

Key design choices based on failure analysis:
1. SIMPLE entry: Donchian breakout ONLY in trend regime
2. HMA(48) 1d-aligned for trend direction (smoother than HMA(21))
3. Choppiness regime filter to avoid whipsaws in ranges
4. Volume confirmation (20-day avg) to filter false breakouts
5. 2x ATR stoploss via signal→0
6. Discrete sizing: 0.30 (breakout) / 0.25 (normal)

Why this should work in BOTH bull AND bear:
- Bull: Price breaks above Donchian + HMA up + volume spike = strong momentum
- Bear: Price breaks below Donchian + HMA down = continuation short
- Range: Choppiness filter prevents trading chop

Target: Sharpe>0.6, trades 75-200 train (19-50/year), DD>-35%
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_chop_vol_1d_v5"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures if market is trending or ranging"""
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - price channel breakout"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA and align to 4h
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=48)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume: 20-day average for confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BREAKOUT = 0.30
    SIZE_BASE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if critical indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(vol_sma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        chop = chop_14[i]
        is_trending = chop < 38.2  # Trending regime
        is_range = chop > 61.8     # Ranging regime
        
        # === TREND DIRECTION (1d HMA) ===
        hma_val = hma_1d_aligned[i]
        bullish_trend = close[i] > hma_val
        bearish_trend = close[i] < hma_val
        
        # === DONCHIAN BREAKOUT (price closes outside channel) ===
        # Use previous bar's channel to avoid look-ahead
        prev_upper = donch_upper[i-1] if i > 0 and not np.isnan(donch_upper[i-1]) else 0
        prev_lower = donch_lower[i-1] if i > 0 and not np.isnan(donch_lower[i-1]) else 0
        
        breakout_long = close[i] > prev_upper and prev_upper > 0
        breakout_short = close[i] < prev_lower and prev_lower > 0
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = volume[i] > vol_sma_20[i] * 1.2  # 20% above average
        
        # === ENTRY LOGIC (SIMPLE: breakout + trend + regime + volume) ===
        desired_signal = 0.0
        
        # ONLY trade in trending regime (CHOP < 38.2)
        if is_trending:
            # LONG: Bullish trend + Donchian breakout + volume
            if bullish_trend and breakout_long and vol_confirmed:
                desired_signal = SIZE_BREAKOUT
            
            # SHORT: Bearish trend + Donchian breakout + volume
            elif bearish_trend and breakout_short and vol_confirmed:
                desired_signal = -SIZE_BREAKOUT
        
        # In ranging regime, only take counter-trend setups with volume
        elif is_range:
            # Range bounce from lower band
            if bullish_trend and close[i] < prev_lower and vol_confirmed:
                desired_signal = SIZE_BASE
            
            # Range bounce from upper band  
            elif bearish_trend and close[i] > prev_upper and vol_confirmed:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL ===
        if desired_signal >= SIZE_BREAKOUT * 0.9:
            final_signal = SIZE_BREAKOUT
        elif desired_signal <= -SIZE_BREAKOUT * 0.9:
            final_signal = -SIZE_BREAKOUT
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = close[i] - 2.0 * entry_atr
                else:
                    stop_price = close[i] + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals