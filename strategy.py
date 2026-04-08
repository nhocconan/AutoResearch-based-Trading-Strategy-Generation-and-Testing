#!/usr/bin/env python3
# 12h_1d_price_channel_breakout_volume_confirm_v1
# Hypothesis: 12-hour Donchian channel breakouts confirmed by 1-day volume surge.
# Long when price breaks above 20-period high with 1d volume > 1.5x 20-day average.
# Short when price breaks below 20-period low with 1d volume > 1.5x 20-day average.
# Works in bull markets via breakout continuation and bear markets via breakdown continuation.
# Volume confirmation reduces false breakouts. Target 15-30 trades/year on 12h.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_price_channel_breakout_volume_confirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 12h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get 1d volume data for confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 1d volume moving average (20-period)
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_current_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure Donchian channels are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition: current 1d volume > 1.5x 20-day average
        vol_surge = False
        if vol_ma_20_aligned[i] > 0:
            vol_surge = vol_current_aligned[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below 12-period low (shorter lookback for responsiveness)
            donchian_low_12 = high_series.rolling(window=12, min_periods=12).min().values[i]
            if close[i] < donchian_low_12:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 12-period high
            donchian_high_12 = low_series.rolling(window=12, min_periods=12).max().values[i]
            if close[i] > donchian_high_12:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-period high with volume surge
            if close[i] > donchian_high[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-period low with volume surge
            elif close[i] < donchian_low[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals