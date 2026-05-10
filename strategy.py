#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v3
# Hypothesis: Breakout above 12h Camarilla R1 or below S1 with volume surge and 1d EMA34 trend confirmation.
# Uses discrete position sizing (0.30) to reduce churn, adds minimum holding period (3 bars = 36h) to limit trades.
# Targets 15-30 trades/year. Works in bull/bear by requiring trend alignment, reducing false breakouts.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v3"
timeframe = "12h"
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
    
    # 12h OHLC for Camarilla calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Camarilla levels using previous period's range
    # R1 = Close + 1.1*(High-Low)/12, S1 = Close - 1.1*(High-Low)/12
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_r1 = close_prev + 1.1 * (high_prev - low_prev) / 12
    camarilla_s1 = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20-period = ~10 days of 12h bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Warmup: need Camarilla (needs 1 bar) + EMA34 (34) + volume MA (20)
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
                signals[i] = 0.30
                position = 1
            # Short: Breakdown below Camarilla S1 with volume surge and 1d downtrend
            elif breakdown_s1 and volume_surge and downtrend:
                signals[i] = -0.30
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 3 bars (36 hours)
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
                    signals[i] = 0.30
            elif position == -1:
                # Short exit: price breaks above Camarilla R1 or trend changes
                if close[i] > camarilla_r1[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.30
    
    return signals