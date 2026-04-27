#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume_Spike_v1
Hypothesis: Trade 12h timeframe with weekly CAMARILLA R1/S1 breakouts filtered by weekly trend and volume spikes.
Breakouts at R1/S1 capture strong momentum, filtered by weekly trend and volume confirmation.
Designed for 12-37 trades/year to minimize fee drift. Works in bull via breakouts and bear via breakdowns.
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
    
    # Calculate CAMARILLA levels from 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for CAMARILLA calculation
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # CAMARILLA R1 and S1 levels (breakout levels)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align CAMARILLA levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Weekly EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Volume confirmation: current volume > 2.0 * 4-period average (on 12h data, ~2 days)
    vol_avg = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for volume average and EMA
    start_idx = max(4, 34)
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_34_val = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Require minimum 3 bars since last exit to avoid churn (~1.5 days on 12h)
            if bars_since_exit >= 3:
                # Long: price breaks above R1 with volume confirmation AND above weekly EMA34 (uptrend)
                if close[i] > camarilla_r1_val and vol_conf and close[i] > ema_34_val:
                    signals[i] = size
                    position = 1
                    bars_since_exit = 0
                # Short: price breaks below S1 with volume confirmation AND below weekly EMA34 (downtrend)
                elif close[i] < camarilla_s1_val and vol_conf and close[i] < ema_34_val:
                    signals[i] = -size
                    position = -1
                    bars_since_exit = 0
        elif position == 1:
            # Exit long: price breaks below S1 (opposite level)
            if close[i] < camarilla_s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R1 (opposite level)
            if close[i] > camarilla_r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume_Spike_v1"
timeframe = "12h"
leverage = 1.0