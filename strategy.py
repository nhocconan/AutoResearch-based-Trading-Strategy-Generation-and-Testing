#!/usr/bin/env python3
"""
6h Weekly Pivot Direction + Donchian(20) Breakout + Volume Spike
Hypothesis: Weekly pivot levels from 1w timeframe establish institutional support/resistance.
6h Donchian(20) breakouts in direction of weekly pivot bias capture momentum with institutional alignment.
Volume confirmation (>1.8x 20-period MA) filters false breakouts. Designed for 6h timeframe with
50-150 total trades over 4 years, working in both bull and bear via weekly pivot filter and volume.
"""

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
    
    # Get 1w data for weekly pivot calculation (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least 5 weeks for reasonable pivot calc
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # P = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot for each week using prior week's data (shifted by 1)
    pivot = np.full(len(df_1w), np.nan)
    r1 = np.full(len(df_1w), np.nan)
    s1 = np.full(len(df_1w), np.nan)
    r2 = np.full(len(df_1w), np.nan)
    s2 = np.full(len(df_1w), np.nan)
    r3 = np.full(len(df_1w), np.nan)
    s3 = np.full(len(df_1w), np.nan)
    
    for i in range(1, len(df_1w)):
        # Use prior week's OHLC
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        
        p = (ph + pl + pc) / 3.0
        pivot[i] = p
        r1[i] = 2 * p - pl
        s1[i] = 2 * p - ph
        r2[i] = p + (ph - pl)
        s2[i] = p - (ph - pl)
        r3[i] = ph + 2 * (p - pl)
        s3[i] = pl - 2 * (ph - p)
    
    # Align weekly pivot levels to 6h timeframe (completed week only)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 20-period volume MA for volume spike confirmation (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Calculate ATR(14) for stoploss (6h)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    # Calculate Donchian channels (20-period) (6h)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for weekly pivot (with 1-week lag), volume MA, ATR, and Donchian
    start_idx = max(20, 14)  # Weekly pivot alignment handles its own lag
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(atr[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        
        # Weekly pivot bias: price above/below pivot determines long/short bias
        bullish_bias = curr_close > pivot_val
        bearish_bias = curr_close < pivot_val
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        volume_confirm = curr_volume > 1.8 * vol_ma
        
        if position == 0:
            # Look for breakout signals in direction of weekly pivot bias
            # Long: price breaks above Donchian high with volume confirmation in bullish bias
            long_breakout = (curr_close > donch_high) and volume_confirm and bullish_bias
            # Short: price breaks below Donchian low with volume confirmation in bearish bias
            short_breakout = (curr_close < donch_low) and volume_confirm and bearish_bias
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Stoploss: 2 * ATR below entry
            stop_loss = entry_price - 2.0 * atr_val
            # Exit conditions: price closes below Donchian low OR stoploss hit OR weekly pivot bias turns bearish
            if curr_close < donch_low or curr_close < stop_loss or curr_close < pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Stoploss: 2 * ATR above entry
            stop_loss = entry_price + 2.0 * atr_val
            # Exit conditions: price closes above Donchian high OR stoploss hit OR weekly pivot bias turns bullish
            if curr_close > donch_high or curr_close > stop_loss or curr_close > pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Direction_DonchianBreakout_VolumeSpike"
timeframe = "6h"
leverage = 1.0