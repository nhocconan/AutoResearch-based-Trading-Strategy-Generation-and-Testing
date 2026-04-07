#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian20 Volume + 1d EMA Trend Filter
# Hypothesis: Breakout of 12h Donchian(20) channel with volume confirmation
# and daily EMA(20) trend filter. Works in both bull and bear markets by
# trading breakouts in direction of daily trend. Target: 12-37 trades/year
# to minimize fee drag on 12h timeframe.

name = "12h_donchian20_volume_1d_ema_trend_v1"
timeframe = "12h"
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
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) - using rolling window on 12h data
    # We need to calculate Donchian on 12h data directly
    donch_period = 20
    highest_high = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    lowest_low = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # Daily EMA(20) for trend filter
    close_daily = df_daily['close'].values
    ema_20_daily = pd.Series(close_daily).ewm(span=20, adjust=False).mean().values
    ema_20_12h = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    # Volume filter: 12h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donch_period, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_20_12h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend changes
            if close[i] < lowest_low[i] or close[i] < ema_20_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend changes
            if close[i] > highest_high[i] or close[i] > ema_20_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of daily EMA trend with volume
            if vol_ok:
                if close[i] > ema_20_12h[i]:  # Uptrend
                    if high[i] > highest_high[i]:  # Break above Donchian high
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if low[i] < lowest_low[i]:  # Break below Donchian low
                        position = -1
                        signals[i] = -0.25
    
    return signals