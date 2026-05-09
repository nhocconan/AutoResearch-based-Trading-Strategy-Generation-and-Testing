#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily 20-period Donchian Channel Breakout with Weekly Trend and Volume Spike
# Uses daily Donchian high/low breakouts with weekly EMA10 trend alignment and volume confirmation.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend with volume).
# Target: 7-25 trades per year (30-100 total over 4 years) to minimize fee drag.
name = "1d_Donchian20_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Weekly EMA10 for trend filter
    ema10_weekly = pd.Series(df_weekly['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_aligned = align_htf_to_ltf(prices, df_weekly, ema10_weekly)
    
    # Daily Donchian Channel (20-period)
    # Calculate highest high and lowest low over last 20 days
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 days for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema10_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: Break above Donchian high with uptrend and volume spike
            if close[i] > donchian_high[i] and close[i] > ema10_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with downtrend and volume spike
            elif close[i] < donchian_low[i] and close[i] < ema10_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Donchian low OR trend turns down
            if close[i] < donchian_low[i] or close[i] < ema10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above Donchian high OR trend turns up
            if close[i] > donchian_high[i] or close[i] > ema10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals