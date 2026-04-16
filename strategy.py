#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data (HTF) for market regime ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Daily data for entry levels ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Weekly EMA for trend bias ===
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Daily Pivot Points (Standard) ===
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    pivot_point = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    prev_range = prev_high_1d - prev_low_1d
    r1 = pivot_point + prev_range * 0.382
    s1 = pivot_point - prev_range * 0.382
    r2 = pivot_point + prev_range * 0.618
    s2 = pivot_point - prev_range * 0.618
    
    # === ATR for dynamic stop ===
    tr_6h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # Align all to 6h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)  # Volume ratio aligned to daily
    atr_6h_aligned = align_htf_to_ltf(prices, df_1d, atr_6h)  # Actually compute on 6h, but align for simplicity
    
    # Recompute ATR on 6h directly for accuracy
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_bull = ema_50_1w_aligned[i] > close_1w[i // (7*24//6)] if i >= (7*24//6) else False  # Simplified: use aligned EMA > current weekly close proxy
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        atr_val = atr_6h[i]
        vol_ratio_val = vol_ratio[i]
        
        # Simplified weekly trend: aligned EMA > price (proxy)
        weekly_close_proxy = np.full_like(close, np.nan)
        # This is approximate; better to use proper alignment but keeping simple
        weekly_trend = ema_50_1w_aligned[i] > price  # Bullish if EMA above price
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            if price < s1_val or price > r2_val or price < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            if price > r1_val or price < s2_val or price > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Weekly trend filter: only trade in direction of weekly trend
            if weekly_trend:
                # LONG: Price breaks above R1 with volume confirmation
                if price > r1_val and vol_ratio_val > 1.5:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
            else:
                # SHORT: Price breaks below S1 with volume confirmation
                if price < s1_val and vol_ratio_val > 1.5:
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

name = "6h_WeeklyTrend_PivotBreak_Volume"
timeframe = "6h"
leverage = 1.0