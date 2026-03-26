#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla Pivot Zone + Volume Spike + Choppiness (Relaxed)

HYPOTHESIS: The proven Camarilla pattern from DB (Sharpe=1.47) works when we
enter when price touches or crosses S3/R3 zones (not just "near"). 
Relax the distance filter from 0.5 ATR to 3 ATR to capture more setups.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Camarilla S3/R3 are derived from previous range, work in all markets
- Long at S3 in bull, short at R3 in bear
- Range markets: mean-revert between S3/R3

KEY SIMPLIFICATION: Only 3 conditions:
1. Price touches S3 or R3 (within 3 ATR for more triggers)
2. Volume spike > 1.5x 20-avg
3. Choppiness < 55 (trending environment)

TARGET: 75-150 total trades over 4 years (proven pattern).
Reference: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (95tr, Sharpe=1.47)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_s3r3_vol_chop_relaxed_v1"
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
    """Choppiness Index - CHOP < 55 = trending (allow trades)"""
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

def calculate_camarilla_s3r3(prev_high, prev_low, prev_close):
    """
    Camarilla S3 and R3 pivot levels
    S3 = close - (high - low) * 1.1 / 4
    R3 = close + (high - low) * 1.1 / 4
    """
    n = len(prev_high)
    s3 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        high_low_range = prev_high[i] - prev_low[i]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i]
        s3[i] = close - high_low_range * 1.1 / 4
        r3[i] = close + high_low_range * 1.1 / 4
    
    return s3, r3

def calculate_ema(close, span):
    """Exponential Moving Average"""
    return pd.Series(close).ewm(span=span, min_periods=span, adjust=False).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate Camarilla S3/R3 from 1d
    s3_raw, r3_raw = calculate_camarilla_s3r3(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 4h
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_raw)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # EMA for trend confirmation
    ema_8 = calculate_ema(close, 8)
    ema_21 = calculate_ema(close, 21)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Warmup
    warmup = 60
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    stop_price = 0.0
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
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
        
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop = chop_14[i]
        is_trending = chop < 55.0
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === PIVOT LEVELS ===
        s3 = s3_aligned[i]
        r3 = r3_aligned[i]
        
        # Price distance to pivot levels (as % of ATR)
        if atr_14[i] > 0:
            dist_to_s3 = (close[i] - s3) / atr_14[i]
            dist_to_r3 = (r3 - close[i]) / atr_14[i]
        else:
            dist_to_s3 = 999
            dist_to_r3 = 999
        
        # Price crossing pivot levels
        crossed_s3_up = close[i] >= s3 and (i == 0 or close[i-1] < s3)
        crossed_s3_down = close[i] <= s3 and (i == 0 or close[i-1] > s3)
        crossed_r3_up = close[i] >= r3 and (i == 0 or close[i-1] < r3)
        crossed_r3_down = close[i] <= r3 and (i == 0 or close[i-1] > r3)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_trending:
            # LONG: Price touching/crossing S3 + bullish bias + volume
            if (crossed_s3_down or (dist_to_s3 >= -3.0 and dist_to_s3 <= 1.0)) and price_above_1d_hma:
                if vol_spike:
                    desired_signal = SIZE
            
            # SHORT: Price touching/crossing R3 + bearish bias + volume
            if (crossed_r3_up or (dist_to_r3 >= -3.0 and dist_to_r3 <= 1.0)) and not price_above_1d_hma:
                if vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK ===
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
        
        # === TAKE PROFIT at opposite pivot ===
        tp_triggered = False
        if in_position and position_side > 0:
            if not np.isnan(r3) and high[i] >= r3:
                tp_triggered = True
        
        if in_position and position_side < 0:
            if not np.isnan(s3) and low[i] <= s3:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals