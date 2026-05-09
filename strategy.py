#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike
# Works in bull: breakouts capture momentum. Works in bear: EMA34 filter avoids shorts in strong downtrends,
# only allowing longs when price > EMA34 (bullish bias) and shorts when price < EMA34 (bearish bias).
# Volume spike confirms breakout legitimacy. Target: 20-40 trades/year to avoid fee drag.
name = "4h_Donchian20_EMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    # Use pandas rolling for efficiency
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_avg_1d_4h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_4h[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_avg_1d_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average 1d volume
        vol_ok = volume[i] > vol_avg_1d_4h[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above Donchian high, above 1d EMA34, with volume
            if close[i] > donchian_high[i] and close[i] > ema34_1d_4h[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low, below 1d EMA34, with volume
            elif close[i] < donchian_low[i] and close[i] < ema34_1d_4h[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or closes below 1d EMA34
            if close[i] < donchian_low[i] or close[i] < ema34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or closes above 1d EMA34
            if close[i] > donchian_high[i] or close[i] > ema34_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals