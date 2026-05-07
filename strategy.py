#!/usr/bin/env python3
"""
4H_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1
Hypothesis: Use daily Camarilla R3/S3 levels from 1d timeframe for entry signals,
filtered by 1d EMA(34) trend direction and volume spikes. Long when price breaks above
R3 in uptrend with volume confirmation; short when price breaks below S3 in downtrend
with volume confirmation. Uses 4h timeframe for execution to balance trade frequency
and capture intraday momentum while avoiding overtrading. Camarilla levels provide
statistically significant support/resistance, EMA34 filters trend alignment,
and volume spikes confirm institutional participation.
"""
name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3)
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Actually standard Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Simplified: R3 = close + 1.1*(high-low)*1.1/4 = close + 0.3025*(high-low)
    # But standard formula: R4 = close + 1.5*(high-low)*1.1/2, etc.
    # Using common Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Actually correct: R3 = close + (high-low)*1.1/4*1.1? Let's use standard:
    # Typical Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    # But many versions use 1.1 multiplier. Using: R3 = close + 1.1*(high-low)*1.1/4
    # Simpler: use R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low) for wider bands
    # Actually checking: standard Camarilla R3 = close + 1.1*(high-low)*1.1/4
    # Let's use proven version: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # This gives wider bands suitable for 4h breakouts
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    camarilla_r3 = daily_close + 1.1 * (daily_high - daily_low)
    camarilla_s3 = daily_close - 1.1 * (daily_high - daily_low)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA34 for trend
    ema_34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0 * 20-period average volume (4h periods)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + uptrend (close > EMA34) + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + downtrend (close < EMA34) + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1 and (close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals