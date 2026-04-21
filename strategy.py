#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_1wTrend_VolumeConfirm_ATRStop_v1
Hypothesis: Weekly pivot breakout with weekly EMA20 trend filter and volume spike confirmation. Designed for low trade frequency (~10-30/year) to minimize fee drag and work in both bull and bear markets by capturing significant breakouts with strong momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === Weekly pivot points (standard calculation) ===
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3.0
    r1_w = 2 * pivot_w - low_w
    s1_w = 2 * pivot_w - high_w
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly pivots to daily
    pivot_w_aligned = align_htf_to_ltf(prices, df_1w, pivot_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # === Weekly EMA20 trend filter ===
    ema20_w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_w_aligned = align_htf_to_ltf(prices, df_1w, ema20_w)
    
    # === Weekly volume average (10-period) for spike detection ===
    volume_w = df_1w['volume'].values
    vol_ma_w = pd.Series(volume_w).rolling(window=10, min_periods=10).mean().values
    vol_ma_w[np.isnan(vol_ma_w)] = 1.0
    vol_ratio_w = volume_w / vol_ma_w
    vol_ratio_w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_w)
    
    # === ATR for dynamic stoploss (14-period on daily) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(pivot_w_aligned[i]) or np.isnan(r1_w_aligned[i]) or 
            np.isnan(s1_w_aligned[i]) or np.isnan(r2_w_aligned[i]) or 
            np.isnan(s2_w_aligned[i]) or np.isnan(ema20_w_aligned[i]) or
            np.isnan(vol_ratio_w_aligned[i]) or np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        pivot = pivot_w_aligned[i]
        r1 = r1_w_aligned[i]
        s1 = s1_w_aligned[i]
        r2 = r2_w_aligned[i]
        s2 = s2_w_aligned[i]
        trend = ema20_w_aligned[i]
        vol_spike = vol_ratio_w_aligned[i]
        atr_val = atr_14[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above weekly EMA20
            if price_close > r1 and vol_spike > 2.0 and price_close > trend:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below S1 with volume spike and below weekly EMA20
            elif price_close < s1 and vol_spike > 2.0 and price_close < trend:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
                lowest_since_entry = price_close
        
        elif position != 0:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Trailing stop: 2.5 * ATR below highest since entry
                if price_close < highest_since_entry - 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Trailing stop: 2.5 * ATR above lowest since entry
                if price_close > lowest_since_entry + 2.5 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_WeeklyPivot_Breakout_1wTrend_VolumeConfirm_ATRStop_v1"
timeframe = "1d"
leverage = 1.0