#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_VolumeSpike_1wTrend_v2
Hypothesis: Trade 1d Camarilla R3/S3 breakouts in the direction of weekly EMA50 trend with volume confirmation.
Uses tighter volume filter (3.0 * ATR) and adds a minimum hold period of 2 bars to reduce overtrading.
Only long when price breaks above Camarilla R3 AND weekly close > weekly EMA50 AND volume > 3.0 * ATR1d.
Only short when price breaks below Camarilla S3 AND weekly close < weekly EMA50 AND volume > 3.0 * ATR1d.
Discrete sizing 0.25 to limit fee drag. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    c_1w = df_1w['close'].values
    ema_50_1w = pd.Series(c_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla levels (R3/S3)
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla R3 = C + (H-L)*1.1/4
    # Camarilla S3 = C - (H-L)*1.1/4
    camarilla_r3_1d = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_s3_1d = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # Align daily Camarilla levels to 1d timeframe (no shift needed)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Calculate ATR for volume confirmation (using 1d data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track bars in position for minimum hold
    
    # Start index: need warmup for EMA50 and ATR
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume confirmation: current volume > 3.0 * ATR (tighter filter)
        volume_confirm = volume[i] > 3.0 * atr[i]
        
        # Determine weekly trend from EMA50
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, c_1w)[i]
        if np.isnan(weekly_close_aligned):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
            
        if weekly_close_aligned > ema_50_aligned[i]:
            weekly_trend = 'bullish'  # only allow longs
        elif weekly_close_aligned < ema_50_aligned[i]:
            weekly_trend = 'bearish'  # only allow shorts
        else:
            weekly_trend = 'neutral'  # no trades in neutral zone
        
        if position == 0:
            bars_since_entry = 0
            # Long setup: price breaks above Camarilla R3 AND volume confirm AND bullish weekly trend
            long_setup = (close[i] > camarilla_r3_aligned[i]) and volume_confirm and (weekly_trend == 'bullish')
            
            # Short setup: price breaks below Camarilla S3 AND volume confirm AND bearish weekly trend
            short_setup = (close[i] < camarilla_s3_aligned[i]) and volume_confirm and (weekly_trend == 'bearish')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            bars_since_entry += 1
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Camarilla S3 OR weekly trend turns bearish OR min hold (2 bars) + adverse move
            if bars_since_entry >= 2:
                if (close[i] < camarilla_s3_aligned[i]) or (weekly_trend == 'bearish'):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        elif position == -1:
            bars_since_entry += 1
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Camarilla R3 OR weekly trend turns bullish OR min hold (2 bars) + adverse move
            if bars_since_entry >= 2:
                if (close[i] > camarilla_r3_aligned[i]) or (weekly_trend == 'bullish'):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "1d_Camarilla_Pivot_VolumeSpike_1wTrend_v2"
timeframe = "1d"
leverage = 1.0