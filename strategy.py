#!/usr/bin/env python3
"""
Experiment #006: 4h Donchian Breakout + Volume Spike + 1d HMA Trend

HYPOTHESIS: Donchian breakout from previous day's range captures institutional 
momentum moves. Volume spike confirms smart money involvement. 1d HMA filters 
counter-trend trades. ATR stoploss prevents blowups.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Break above prev_day_high + volume + uptrend = strong continuation
- Bear: Break below prev_day_low + volume + downtrend = strong continuation  
- Range: False breakouts get stopped out quickly by ATR

TARGET: 75-150 total trades over 4 years (proven pattern from DB).
DB reference: mtf_4h_hma_donchian_volume_rsi_12h_atr_v1 (Sharpe=1.382, 95 trades)

KEY DESIGN (tight entries = fewer trades = less fee drag):
1. Price BREAKS prev_day_high/Low (not just "near" it)
2. Volume confirms (>1.8x 20-avg)
3. 1d HMA confirms trend direction
4. ATR stoploss (2x) + take profit at opposite pivot
5. SIZE = 0.25 (conservative)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_1d_hma_v1"
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

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """
    Camarilla pivot levels (symmetric above/below close)
    S1/R1 = close ± range/12
    S2/R2 = close ± range/6
    S3/R3 = close ± range/4
    S4/R4 = close ± range/2
    """
    n = len(prev_high)
    pivots = {
        's3': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        high_low_range = prev_high[i] - prev_low[i]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i]
        pivots['s3'][i] = close - high_low_range / 4
        pivots['r3'][i] = close + high_low_range / 4
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend (aligned to 4h, shifted by 1 to avoid lookahead)
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # 1d prev high/low for Donchian breakout (aligned to 4h)
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['high'].values)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['low'].values)
    
    # 1d Camarilla S3/R3 for take profit (aligned to 4h)
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    
    # 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA (20-period on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup (need 4h bars for indicators to populate)
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if 1d indicators not aligned
        if np.isnan(hma_1d_aligned[i]) or np.isnan(prev_high_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # 1d trend: price above HMA21 = bullish, below = bearish
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_aligned[i]
        
        # Previous day's high/low (from aligned 1d)
        prev_day_high = prev_high_1d_aligned[i]
        prev_day_low = prev_low_1d_aligned[i]
        
        if np.isnan(prev_day_high) or np.isnan(prev_day_low):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Volume confirmation: STRICT (1.8x to reduce false breakouts)
        vol_spike = vol_ratio[i] > 1.8
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Break above prev_day_high + volume + uptrend
        if price_above_1d_hma:
            # Breakout: high exceeds previous day's high
            if high[i] > prev_day_high:
                if vol_spike:
                    desired_signal = SIZE
        
        # SHORT: Break below prev_day_low + volume + downtrend
        if price_below_1d_hma:
            # Breakdown: low exceeds previous day's low
            if low[i] < prev_day_low:
                if vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS (ATR-based) ===
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
        
        # === TAKE PROFIT at opposite 1d pivot ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            # TP at R3 (1d resistance)
            r3 = r3_aligned[i]
            if not np.isnan(r3) and high[i] >= r3:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at S3 (1d support)
            s3 = s3_aligned[i]
            if not np.isnan(s3) and low[i] <= s3:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New entry or reversal
                in_position = True
                position_side = int(np.sign(desired_signal))
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
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals