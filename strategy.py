#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with weekly EMA200 trend filter and volume spike
# Uses Elder Ray on 6h timeframe: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and weekly EMA200 uptrend, short when Bear Power < 0 and weekly EMA200 downtrend.
# Volume confirmation (> 1.5x 20-period average) reduces false signals.
# Designed to capture institutional buying/selling pressure with trend filter for bull/bear markets.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_WeeklyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA200 trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend
    close_weekly = df_weekly['close'].values
    ema200_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 200:
        ema200_weekly[199] = np.mean(close_weekly[:200])
        for i in range(200, len(close_weekly)):
            ema200_weekly[i] = (close_weekly[i] * 2 + ema200_weekly[i-1] * 198) / 200
    
    # Calculate 6h EMA13 for Elder Ray
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema13[i] = (close[i] * 2 + ema13[i-1] * 11) / 13
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Calculate 6h volume average for volume spike
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA200 to 6h timeframe
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 13)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema200_weekly_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        weekly_uptrend = close[i] > ema200_weekly_aligned[i]
        weekly_downtrend = close[i] < ema200_weekly_aligned[i]
        vol_spike = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Elder Ray signal with trend and volume confirmation
            if bull_power[i] > 0 and weekly_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            elif bear_power[i] < 0 and weekly_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power turns positive or trend fails or volume drops
            if bear_power[i] > 0 or not weekly_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power turns negative or trend fails or volume drops
            if bull_power[i] < 0 or not weekly_downtrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals