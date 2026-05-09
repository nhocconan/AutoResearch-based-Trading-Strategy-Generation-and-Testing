#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout (20) with Daily Trend Filter and Volume Spike
# Uses daily Donchian upper/lower bands for breakout signals, daily EMA34 for trend alignment,
# and volume spike for confirmation. Works in bull markets (breakouts with trend) and bear markets
# (breakouts against trend with volume). Designed for 20-40 trades/year to avoid fee drag.
name = "4h_DonchianBreakout_DailyTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian bands and EMA trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(df_daily['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Daily Donchian channels (20-period)
    donchian_high = pd.Series(df_daily['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_daily['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h
    donchian_high_4h = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_daily, donchian_low)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_4h[i]) or np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Break above daily Donchian high with daily uptrend and volume spike
            if close[i] > donchian_high_4h[i] and close[i] > ema34_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below daily Donchian low with daily downtrend and volume spike
            elif close[i] < donchian_low_4h[i] and close[i] < ema34_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below daily Donchian low OR daily trend turns down
            if close[i] < donchian_low_4h[i] or close[i] < ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above daily Donchian high OR daily trend turns up
            if close[i] > donchian_high_4h[i] or close[i] > ema34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals