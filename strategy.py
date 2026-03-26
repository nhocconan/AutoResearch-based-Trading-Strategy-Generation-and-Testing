#!/usr/bin/env python3
"""
Experiment #021: 12h Camarilla + Volume + 1w Trend

HYPOTHESIS:
- Camarilla pivot levels from 1w capture institutional S/R that work in all markets
- 12h timeframe reduces overtrading vs 4h (fewer bars = fewer false signals)
- 1w HMA trend filter prevents countertrend entries
- Volume spike confirms institutional involvement
- ATR stoploss protects against false breakouts

WHY 12h:
- DB shows 4h Camarilla works (Sharpe 1.47), but many 4h attempts overtrade
- 12h = 2x fewer bars = naturally tighter trade frequency
- 1w data for pivots maintains institutional-grade levels

TARGET: 75-150 total trades over 4 years (proven range from DB)
DB reference: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (Sharpe=1.471, 95 trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_1w_v1"
timeframe = "12h"
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
    """Choppiness Index - measures market choppiness"""
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

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """Camarilla pivot levels"""
    n = len(prev_high)
    pivots = {
        's3': np.full(n, np.nan, dtype=np.float64),
        's4': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
        'r4': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        high_low_range = prev_high[i] - prev_low[i]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i]
        
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['s4'][i] = close - high_low_range * 1.1 / 2
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
        pivots['r4'][i] = close + high_low_range * 1.1 / 2
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data ONCE for trend and Camarilla
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Camarilla pivots from 1w
    cam_pivots = calculate_camarilla_pivots(
        df_1w['high'].values,
        df_1w['low'].values,
        df_1w['close'].values
    )
    
    # Align pivots to 12h
    s3_aligned = align_htf_to_ltf(prices, df_1w, cam_pivots['s3'])
    s4_aligned = align_htf_to_ltf(prices, df_1w, cam_pivots['s4'])
    r3_aligned = align_htf_to_ltf(prices, df_1w, cam_pivots['r3'])
    r4_aligned = align_htf_to_ltf(prices, df_1w, cam_pivots['r4'])
    
    # 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 30
    
    for i in range(warmup, n):
        # Validations
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(vol_ratio[i]):
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
        
        # Regime: trending if chop < 55
        is_trending = chop_14[i] < 55.0
        
        # 1w trend: above HMA = bullish
        price_above_1w_hma = close[i] > hma_1w_aligned[i] if not np.isnan(hma_1w_aligned[i]) else True
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Camarilla levels
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        
        # Distance to levels in ATR units
        if atr_14[i] > 0:
            dist_s3 = (close[i] - s3) / atr_14[i]
            dist_s4 = (close[i] - s4) / atr_14[i] if not np.isnan(s4) else 999
            dist_r3 = (r3 - close[i]) / atr_14[i]
            dist_r4 = (r4 - close[i]) / atr_14[i] if not np.isnan(r4) else 999
        else:
            dist_s3 = dist_s4 = dist_r3 = dist_r4 = 999
        
        # Entry logic
        desired_signal = 0.0
        
        if is_trending:
            # LONG: At S3/S4 support with bullish 1w trend + volume
            if dist_s3 > -0.5 and dist_s3 < 2.0 and price_above_1w_hma:
                if vol_spike:
                    desired_signal = SIZE
                else:
                    desired_signal = SIZE * 0.5  # partial without vol
            
            if dist_s4 > -0.5 and dist_s4 < 2.0 and price_above_1w_hma:
                if vol_spike:
                    desired_signal = SIZE
            
            # SHORT: At R3/R4 resistance with bearish 1w trend + volume
            if dist_r3 > -0.5 and dist_r3 < 2.0 and not price_above_1w_hma:
                if vol_spike:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE * 0.5
            
            if dist_r4 > -0.5 and dist_r4 < 2.0 and not price_above_1w_hma:
                if vol_spike:
                    desired_signal = -SIZE
        
        # Stoploss check
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # Take profit at opposite pivot
        if in_position and position_side > 0:
            if (not np.isnan(r3) and high[i] >= r3) or (not np.isnan(r4) and high[i] >= r4):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if (not np.isnan(s3) and low[i] <= s3) or (not np.isnan(s4) and low[i] <= s4):
                desired_signal = 0.0
        
        # Update position
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
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
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals