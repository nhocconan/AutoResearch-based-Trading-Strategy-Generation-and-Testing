#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + Volume + 1d EMA Trend Filter
# Hypothesis: Breakouts of 4h Donchian channels with volume confirmation and
# daily EMA trend filter work in both bull and bear markets by capturing
# momentum in the direction of higher timeframe trend. Daily EMA filter
# avoids counter-trend trades during reversals.
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.

name = "4h_donchian20_volume_1d_ema_trend_v1"
timeframe = "4h"
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
    
    # Calculate 4h Donchian channels (20-period)
    donch_len = 20
    donch_high = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    donch_low = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Daily EMA(20) for trend filter
    close_daily = df_daily['close'].values
    ema_20_daily = pd.Series(close_daily).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    # Volume filter: 4h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donch_len, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_20_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend changes
            if low[i] <= donch_low[i] or close[i] < ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend changes
            if high[i] >= donch_high[i] or close[i] > ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Breakout in direction of daily EMA trend with volume
            if vol_ok:
                if close[i] > ema_20_4h[i]:  # Uptrend
                    if high[i] > donch_high[i]:  # Break above Donchian high
                        position = 1
                        signals[i] = 0.30
                else:  # Downtrend
                    if low[i] < donch_low[i]:  # Break below Donchian low
                        position = -1
                        signals[i] = -0.30
    
    return signals