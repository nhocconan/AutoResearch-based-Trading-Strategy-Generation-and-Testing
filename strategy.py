#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Breakout with Weekly EMA Trend Filter and Volume Confirmation
# Hypothesis: Breakouts of daily Donchian channels in direction of weekly EMA trend
# with volume confirmation provide persistent edge across bull/bear markets.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "1d_donchian_breakout_weekly_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    close_weekly = df_weekly['close'].values
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False).mean().values
    ema_20_daily = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Daily Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = np.roll(high_roll, 1)  # Previous day's upper channel
    lower_channel = np.roll(low_roll, 1)   # Previous day's lower channel
    
    # Volume filter: daily volume > 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_20_daily[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price reaches lower channel or trend changes
            if low[i] <= lower_channel[i] or close[i] < ema_20_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches upper channel or trend changes
            if high[i] >= upper_channel[i] or close[i] > ema_20_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of weekly EMA trend with volume confirmation
            if vol_ok:
                if close[i] > ema_20_daily[i]:  # Uptrend
                    if high[i] > upper_channel[i]:
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if low[i] < lower_channel[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals