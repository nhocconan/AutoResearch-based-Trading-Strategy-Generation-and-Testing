# 6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter
# Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance.
# Breakouts above R1 or below S1 with volume confirmation indicate trend continuation.
# Works in both bull and bear markets by following breakouts from key daily levels.
# Uses 6h timeframe with 12h trend filter to reduce whipsaw.
# Target: 50-150 total trades over 4 years (12-37/year).

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
    
    # === 12h data (trend filter) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # === 1d data (for Camarilla pivots) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 12h EMA34 (trend filter) ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === Daily Camarilla pivot levels (R1, S1) ===
    # Pivot point = (H + L + C) / 3
    # R1 = P + 1.1 * (H - L) / 2
    # S1 = P - 1.1 * (H - L) / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + 1.1 * (high_1d - low_1d) / 2.0
    s1_1d = pivot_1d - 1.1 * (high_1d - low_1d) / 2.0
    
    # Align Camarilla levels to 6h timeframe (with delay for completed daily bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 6h volume ratio for confirmation ===
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_6h = volume_6h / vol_ma_20_6h
    
    # === 6h ATR (14-period) for stop loss ===
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_14_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position and entry price for stop loss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ratio_6h[i]) or
            np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_trend = ema_34_12h_aligned[i]
        vol_ratio = vol_ratio_6h[i]
        atr = atr_14_6h[i]
        
        # === STOP LOSS LOGIC ===
        if position == 1:  # Long position
            if price < entry_price - 2.5 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1:  # Short position
            if price > entry_price + 2.5 * atr:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit: price closes below S1 or trend reverses (below EMA34)
            if price < s1 or price < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1:  # Short position
            # Exit: price closes above R1 or trend reverses (above EMA34)
            if price > r1 or price > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Break above R1 with volume confirmation, in uptrend (above EMA34)
            if price > r1 and vol_ratio > 2.0 and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # SHORT: Break below S1 with volume confirmation, in downtrend (below EMA34)
            elif price < s1 and vol_ratio > 2.0 and price < ema_trend:
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

name = "6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0