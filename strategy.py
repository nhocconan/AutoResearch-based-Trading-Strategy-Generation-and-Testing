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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 1w data (HTF for trend and context) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # === 1d data (HTF for pivot levels) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 6h ATR(14) for volatility and stoploss ===
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_14_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_14_6h)
    
    # === 1w EMA50 for trend filter ===
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d Pivot Points (classic) ===
    # Previous day's pivot levels
    pp_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = 2 * pp_1d - low_1d
    s1_1d = 2 * pp_1d - high_1d
    r2_1d = pp_1d + (high_1d - low_1d)
    s2_1d = pp_1d - (high_1d - low_1d)
    r3_1d = high_1d + 2 * (pp_1d - low_1d)
    s3_1d = low_1d - 2 * (high_1d - pp_1d)
    
    # Align pivot levels to 6h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 6h volume ratio for confirmation ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_14_6h_aligned[i]) or
            np.isnan(vol_ratio_6h[i]) or
            np.isnan(pp_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(r2_1d_aligned[i]) or
            np.isnan(s2_1d_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend = ema_50_1w_aligned[i]
        atr = atr_14_6h_aligned[i]
        vol_ratio = vol_ratio_6h[i]
        pp = pp_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below entry - 2.0 * ATR
            if price < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above entry + 2.0 * ATR
            if price > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below S1 or trend reverses
            if price < s1 or price < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above R1 or trend reverses
            if price > r1 or price > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above R1 with volume, in uptrend (above weekly EMA50)
            if (price > r1 and vol_ratio > 1.5 and price > ema_trend):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Break below S1 with volume, in downtrend (below weekly EMA50)
            elif (price < s1 and vol_ratio > 1.5 and price < ema_trend):
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Pivot_R1S1_WeeklyEMA50_Volume_ATRStop_v1"
timeframe = "6h"
leverage = 1.0