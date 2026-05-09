#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA Crossover with 4h Trend Filter and Volume Spike
# Uses 9/21 EMA crossover for entry timing on 1h, filtered by 4h EMA50 trend direction
# Volume spike confirms institutional participation
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Target: 15-37 trades/year (60-150 over 4 years) to stay within fee limits
name = "1h_EMA9_21_Crossover_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # EMA9 and EMA21 for crossover signals
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_1h = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(ema50_4h_1h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: EMA9 crosses above EMA21 + above 4h EMA50 + volume spike
            if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] and close[i] > ema50_4h_1h[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: EMA9 crosses below EMA21 + below 4h EMA50 + volume spike
            elif ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] and close[i] < ema50_4h_1h[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: EMA9 crosses below EMA21 OR price below 4h EMA50
            if ema9[i] < ema21[i] and ema9[i-1] >= ema21[i-1] or close[i] < ema50_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: EMA9 crosses above EMA21 OR price above 4h EMA50
            if ema9[i] > ema21[i] and ema9[i-1] <= ema21[i-1] or close[i] > ema50_4h_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals