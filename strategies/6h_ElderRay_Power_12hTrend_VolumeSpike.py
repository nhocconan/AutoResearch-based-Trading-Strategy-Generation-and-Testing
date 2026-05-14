#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 12h Trend Filter + Volume Spike
# Elder Ray measures bull/bear power relative to EMA, effective in both bull and bear markets.
# 12h trend filter ensures we trade with higher timeframe momentum.
# Volume spike confirms institutional participation.
# Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag.
name = "6h_ElderRay_Power_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 13-period EMA for Elder Ray (13 is standard)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h EMA50 to 6h
    ema50_12h_6h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_12h_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) + above 12h EMA50 + volume spike
            if bull_power[i] > 0 and close[i] > ema50_12h_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) + below 12h EMA50 + volume spike
            elif bear_power[i] < 0 and close[i] < ema50_12h_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power turns negative OR price below 12h EMA50
            if bear_power[i] < 0 or close[i] < ema50_12h_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power turns positive OR price above 12h EMA50
            if bull_power[i] > 0 or close[i] > ema50_12h_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals