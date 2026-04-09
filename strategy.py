#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_volume_v1
# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long: Price breaks above 20-day high AND weekly close > weekly EMA20 AND volume > 1.5x 20-day avg volume.
# Short: Price breaks below 20-day low AND weekly close < weekly EMA20 AND volume > 1.5x 20-day avg volume.
# Exit: Opposite Donchian breakout (long exits on 20-day low break, short exits on 20-day high break).
# Uses weekly EMA for trend filter to avoid counter-trend trades in bear markets.
# Volume confirmation ensures breakout legitimacy. Target: 7-25 trades/year (30-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v1"
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
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    close_1w_s = pd.Series(close_1w)
    ema_20_1w = close_1w_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Price breaks below 20-day low
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price breaks above 20-day high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above 20-day high AND weekly trend up AND volume confirmed
            if (close[i] > donchian_high[i] and 
                close[i] > ema_20_1w_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below 20-day low AND weekly trend down AND volume confirmed
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_20_1w_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals