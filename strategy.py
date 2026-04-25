#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Trade 12h Camarilla H3/L3 breakouts aligned with weekly EMA50 trend and volume spike (volume > 1.5 * ATR14).
Uses discrete sizing 0.25 to limit fee drag. Target: 12-30 trades/year to avoid fee drag while maintaining edge.
Works in bull/bear via weekly trend filter - only long in uptrend, short in downtrend.
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
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily OHLC for Camarilla levels (H3/L3)
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Camarilla H3 = C + (H-L)*1.1/4
    # Camarilla L3 = C - (H-L)*1.1/4
    camarilla_h3_1d = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_l3_1d = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Calculate ATR for volume confirmation (using 12h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track bars in position for minimum hold
    
    # Start index: need warmup for weekly EMA50 and ATR
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume confirmation: current volume > 1.5 * ATR (tightened to reduce trades)
        volume_confirm = volume[i] > 1.5 * atr[i]
        
        # Determine weekly trend from EMA50
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)[i]
        if np.isnan(weekly_close_aligned):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
            
        if weekly_close_aligned > ema_50_1w_aligned[i]:
            weekly_trend = 'bullish'  # only allow longs
        elif weekly_close_aligned < ema_50_1w_aligned[i]:
            weekly_trend = 'bearish'  # only allow shorts
        else:
            weekly_trend = 'neutral'  # no trades in neutral zone
        
        if position == 0:
            bars_since_entry = 0
            # Long setup: price breaks above Camarilla H3 AND volume confirm AND bullish weekly trend
            long_setup = (close[i] > camarilla_h3_aligned[i]) and volume_confirm and (weekly_trend == 'bullish')
            
            # Short setup: price breaks below Camarilla L3 AND volume confirm AND bearish weekly trend
            short_setup = (close[i] < camarilla_l3_aligned[i]) and volume_confirm and (weekly_trend == 'bearish')
            
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
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = 0.25
            else:
                # Long: hold position
                signals[i] = 0.25
                # Exit: price breaks below Camarilla L3 OR weekly trend turns bearish
                if (close[i] < camarilla_l3_aligned[i]) or (weekly_trend == 'bearish'):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        elif position == -1:
            bars_since_entry += 1
            # Minimum holding period: 2 bars
            if bars_since_entry < 2:
                signals[i] = -0.25
            else:
                # Short: hold position
                signals[i] = -0.25
                # Exit: price breaks above Camarilla H3 OR weekly trend turns bullish
                if (close[i] > camarilla_h3_aligned[i]) or (weekly_trend == 'bullish'):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0