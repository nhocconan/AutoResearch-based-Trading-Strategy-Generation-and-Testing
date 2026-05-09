#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian Breakout (20) with Weekly Trend Filter and Volume Spike
# Uses weekly Donchian upper/lower bands for breakout signals, weekly EMA34 for trend alignment,
# and volume spike for confirmation. Works in bull markets (breakouts with trend) and bear markets
# (breakouts against trend with volume). Designed for 10-25 trades/year to avoid fee drag.
name = "1d_DonchianBreakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian bands and EMA trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema34_weekly = pd.Series(df_weekly['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Weekly Donchian channels (20-period)
    donchian_high = pd.Series(df_weekly['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_weekly['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1d
    donchian_high_1d = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_1d = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d[i]) or np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Break above weekly Donchian high with weekly uptrend and volume spike
            if close[i] > donchian_high_1d[i] and close[i] > ema34_1d[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly Donchian low with weekly downtrend and volume spike
            elif close[i] < donchian_low_1d[i] and close[i] < ema34_1d[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below weekly Donchian low OR weekly trend turns down
            if close[i] < donchian_low_1d[i] or close[i] < ema34_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above weekly Donchian high OR weekly trend turns up
            if close[i] > donchian_high_1d[i] or close[i] > ema34_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals