#!/usr/bin/env python3
name = "1d_WeeklyPivot_R1S1_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly high and low for pivot calculation
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate Camarilla R1 and S1 from weekly data
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = pivot + (weekly_high - weekly_low) * 1.1 / 12
    s1 = pivot - (weekly_high - weekly_low) * 1.1 / 12
    
    # Align weekly levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily EMA(34) for trend filter
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 1.8
            uptrend = ema_34[i] > ema_34[i-1]  # Rising EMA
            
            if close[i] > r1_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and daily downtrend
            elif close[i] < s1_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily breakout of weekly Camarilla R1/S1 levels with trend filter and volume confirmation
# - Weekly Camarilla levels identify key support/resistance from prior week
# - Breakout above R1 or below S1 indicates momentum shift
# - Daily EMA(34) ensures alignment with intermediate-term trend
# - Volume spike (1.8x average) confirms institutional participation
# - Works in bull (buy R1 breakouts in uptrend) and bear (sell S1 breakdowns in downtrend)
# - Position size 0.25 targets 15-25 trades/year, avoiding fee drag
# - Exit at opposite level provides clear risk/reward in trending markets