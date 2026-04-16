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
    
    # === 1d data (HTF for key levels) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate 1d Camarilla pivot levels ===
    # Using previous day's OHLC
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First value
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Camarilla formula
    camarilla_base = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_range = prev_high_1d - prev_low_1d
    
    # Resistance and Support levels
    r3 = camarilla_base + camarilla_range * 1.1 / 4
    s3 = camarilla_base - camarilla_range * 1.1 / 4
    
    # Align to 4h timeframe
    camarilla_base_aligned = align_htf_to_ltf(prices, df_1d, camarilla_base)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 1d EMA34 for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume confirmation (4h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # === Choppiness regime filter (4h) ===
    # Chop = 100 * log(sum(TR, n) / (max(high, n) - min(low, n))) / log(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log(atr_sum / (max_high - min_low + 1e-10)) / np.log(14)
    chop = np.where((max_high - min_low) > 0, chop, 50.0)  # Avoid division by zero
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_34_1d_val = ema_34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        chop_val = chop[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below S3 (stop) or hits R3*1.5 (take profit)
            if price < s3_val or price > r3_val * 1.5:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above R3 (stop) or hits S3*0.5 (take profit)
            if price > r3_val or price < s3_val * 0.5:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R3 with volume AND above 1d EMA34 (uptrend)
            # Only in trending market (chop < 61.8)
            if (price > r3_val) and (price > ema_34_1d_val) and (vol_ratio_val > 2.0) and (chop_val < 61.8):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S3 with volume AND below 1d EMA34 (downtrend)
            # Only in trending market (chop < 61.8)
            elif (price < s3_val) and (price < ema_34_1d_val) and (vol_ratio_val > 2.0) and (chop_val < 61.8):
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

name = "4h_Camarilla_R3_S3_Breakout_Volume_EMA34_ChopFilter"
timeframe = "4h"
leverage = 1.0