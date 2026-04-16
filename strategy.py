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
    
    # === 1d data (HTF for pivot levels) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d ATR(14) for volatility filter ===
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # === 6s EMA34 (trend filter) ===
    ema_34_6h = pd.Series(close_6h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_34_6h)
    
    # === 1d Camarilla pivot levels ===
    # Pivot point = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = PP + (H - L) * 1.1 / 12, S1 = PP - (H - L) * 1.1 / 12
    # R2 = PP + (H - L) * 1.1 / 6, S2 = PP - (H - L) * 1.1 / 6
    # R3 = PP + (H - L) * 1.1 / 4, S3 = PP - (H - L) * 1.1 / 4
    # R4 = PP + (H - L) * 1.1 / 2, S4 = PP - (H - L) * 1.1 / 2
    range_1d = high_1d - low_1d
    r1_1d = pp_1d + range_1d * 1.1 / 12
    s1_1d = pp_1d - range_1d * 1.1 / 12
    r2_1d = pp_1d + range_1d * 1.1 / 6
    s2_1d = pp_1d - range_1d * 1.1 / 6
    r3_1d = pp_1d + range_1d * 1.1 / 4
    s3_1d = pp_1d - range_1d * 1.1 / 4
    r4_1d = pp_1d + range_1d * 1.1 / 2
    s4_1d = pp_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 6h volume ratio for confirmation ===
    vol_ma_10_6h = pd.Series(volume_6h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_6h = volume_6h / vol_ma_10_6h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_6h_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_ratio_6h[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_trend = ema_34_6h_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio = vol_ratio_6h[i]
        
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
            # Exit: price closes below S1 or trend reverses (below EMA34)
            if price < s1_aligned[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit: price closes above R1 or trend reverses (above EMA34)
            if price > r1_aligned[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above R1 with volume, in uptrend (above EMA34)
            # Only trade when volatility is elevated (ATR > 0.3 * ATR mean) to avoid chop
            atr_mean = np.nanmean(atr_14_1d_aligned[max(0, i-50):i+1])
            if (price > r1_aligned[i] and vol_ratio > 1.5 and price > ema_trend and 
                atr > 0.3 * atr_mean):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Break below S1 with volume, in downtrend (below EMA34)
            elif (price < s1_aligned[i] and vol_ratio > 1.5 and price < ema_trend and 
                  atr > 0.3 * atr_mean):
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

name = "6h_Camarilla_R1S1_EMA34_Volume_VolatilityFilter"
timeframe = "6h"
leverage = 1.0