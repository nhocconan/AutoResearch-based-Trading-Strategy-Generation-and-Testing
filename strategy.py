#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Reversal
# Hypothesis: Breakout above 1d Camarilla R1 or below S1 with volume surge and 1d EMA34 trend confirmation.
# Focus on 4h timeframe to reduce trade frequency and improve generalization. Uses discrete sizing (0.25).
# Targets 20-40 trades/year. Works in bull/bear by requiring trend alignment, reducing false breakouts.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Reversal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLC for Camarilla calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla levels using previous day's range
    # R1 = Close + 1.1*(High-Low)/12, S1 = Close - 1.1*(High-Low)/12
    high_prev = np.roll(df_1d['high'].values, 1)
    low_prev = np.roll(df_1d['low'].values, 1)
    close_prev = np.roll(df_1d['close'].values, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_r1 = close_prev + 1.1 * (high_prev - low_prev) / 12
    camarilla_s1 = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20-period = ~20 days of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Warmup: need Camarilla (needs 1 day) + EMA34 (34) + volume MA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from 1d EMA34
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        uptrend = close_1d_aligned[i] > ema_34_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout above Camarilla R1 or breakdown below S1
        breakout_r1 = close[i] > camarilla_r1[i]
        breakdown_s1 = close[i] < camarilla_s1[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Breakout above Camarilla R1 with volume surge and 1d uptrend
            if breakout_r1 and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Camarilla S1 with volume surge and 1d downtrend
            elif breakdown_s1 and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 3 bars (12 hours)
            if bars_since_entry < 3:
                signals[i] = signals[i-1]  # maintain position
                continue
            
            if position == 1:
                # Long exit: price breaks below Camarilla S1 or trend changes
                if close[i] < camarilla_s1[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price breaks above Camarilla R1 or trend changes
                if close[i] > camarilla_r1[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals