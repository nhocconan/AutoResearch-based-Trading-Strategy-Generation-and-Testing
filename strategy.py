#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 12h data (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h ATR (14)
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 1d ATR (14)
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 6h Donchian Channel (20) ===
    highest_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # === 12h Supertrend (ATR=10, mult=3) for trend filter ===
    # Calculate basic upper/lower bands
    hl2_12h = (high_12h + low_12h) / 2
    upper_basic = hl2_12h + 3 * atr_12h
    lower_basic = hl2_12h - 3 * atr_12h
    
    # Initialize final bands
    upper_final = np.full_like(upper_basic, np.nan)
    lower_final = np.full_like(lower_basic, np.nan)
    
    for i in range(len(upper_basic)):
        if np.isnan(upper_basic[i]) or np.isnan(lower_basic[i]):
            continue
        if i == 0:
            upper_final[i] = upper_basic[i]
            lower_final[i] = lower_basic[i]
        else:
            upper_final[i] = upper_basic[i] if (upper_basic[i] < upper_final[i-1] or close_12h[i-1] > upper_final[i-1]) else upper_final[i-1]
            lower_final[i] = lower_basic[i] if (lower_basic[i] > lower_final[i-1] or close_12h[i-1] < lower_final[i-1]) else lower_final[i-1]
    
    # Determine Supertrend direction
    supertrend_dir = np.full_like(close_12h, np.nan)
    for i in range(len(close_12h)):
        if np.isnan(upper_final[i]) or np.isnan(lower_final[i]):
            continue
        if i == 0:
            supertrend_dir[i] = 1 if close_12h[i] > upper_final[i] else -1
        else:
            if supertrend_dir[i-1] == -1 and close_12h[i] > upper_final[i]:
                supertrend_dir[i] = 1
            elif supertrend_dir[i-1] == 1 and close_12h[i] < lower_final[i]:
                supertrend_dir[i] = -1
            else:
                supertrend_dir[i] = supertrend_dir[i-1]
    
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, supertrend_dir)
    
    # === 6d ADX (14) for trend strength ===
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(low_1d)
    tr_1d_adx = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(high_1d[i] - high_1d[i-1], 0) if (high_1d[i] - high_1d[i-1]) > (low_1d[i-1] - low_1d[i]) else 0
        minus_dm[i] = max(low_1d[i-1] - low_1d[i], 0) if (low_1d[i-1] - low_1d[i]) > (high_1d[i] - high_1d[i-1]) else 0
        tr_1d_adx[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - high_1d[i-1]), abs(low_1d[i] - low_1d[i-1]))
    
    tr_1d_adx[0] = high_1d[0] - low_1d[0]
    
    # Smooth with Wilder's smoothing (equivalent to RMA)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = (prev_smooth * (period-1) + current) / period
        for i in range(period, len(data)):
            if np.isnan(result[i-1]) or np.isnan(data[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_1d_adx = wilders_smooth(tr_1d_adx, 14)
    plus_di_14 = 100 * wilders_smooth(plus_dm, 14) / atr_1d_adx
    minus_di_14 = 100 * wilders_smooth(minus_dm, 14) / atr_1d_adx
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = wilders_smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6d Williams %R (14) for overbought/oversold ===
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r[highest_high_14 == lowest_low_14] = -50  # Avoid division by zero
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_12h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(supertrend_dir_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        atr_12h_val = atr_12h_aligned[i]
        supertrend_dir_val = supertrend_dir_aligned[i]
        adx_val = adx_aligned[i]
        williams_r_val = williams_r_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 6h Donchian lower OR ADX weakens (<20) OR Supertrend turns bearish
            if (price < lowest_20[i]) or (adx_val < 20) or (supertrend_dir_val == -1):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 6h Donchian upper OR ADX weakens (<20) OR Supertrend turns bullish
            if (price > highest_20[i]) or (adx_val < 20) or (supertrend_dir_val == 1):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above 6h Donchian upper AND Supertrend bullish AND Williams %R oversold (< -80)
            if (price > highest_20[i]) and (supertrend_dir_val == 1) and (williams_r_val < -80):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below 6h Donchian lower AND Supertrend bearish AND Williams %R overbought (> -20)
            elif (price < lowest_20[i]) and (supertrend_dir_val == -1) and (williams_r_val > -20):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_Supertrend_WilliamsR"
timeframe = "6h"
leverage = 1.0