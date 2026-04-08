#!/usr/bin/env python3
# 1h_donchian_breakout_volume_4h1d
# Hypothesis: Breakout strategy using 4h Donchian channels for direction and 1h volume + Donchian breakout for entry, filtered by 1d trend (price > SMA50). 
# Long when 1h price breaks above 4h Donchian upper (20) with volume > 1.5x average and price > 1d SMA50.
# Short when 1h price breaks below 4h Donchian lower (20) with volume > 1.5x average and price < 1d SMA50.
# Exit when price returns to 4h Donchian middle or volume drops below average.
# Uses 4h for trend direction, 1h for entry timing, 1d for regime filter.
# Target: 15-30 trades/year with strict multi-timeframe confluence.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_donchian_breakout_volume_4h1d"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian Channel (20) for trend direction
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_period = 20
    
    # Calculate 4h Donchian bands
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donch_upper_4h = high_series_4h.rolling(window=donch_period, min_periods=donch_period).max().values
    donch_lower_4h = low_series_4h.rolling(window=donch_period, min_periods=donch_period).min().values
    donch_middle_4h = (donch_upper_4h + donch_lower_4h) / 2
    
    # Align 4h Donchian to 1h timeframe
    donch_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_upper_4h)
    donch_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_lower_4h)
    donch_middle_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_middle_4h)
    
    # 1d SMA50 for regime filter (bull/bear)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_period = 50
    sma_1d = pd.Series(close_1d).rolling(window=sma_period, min_periods=sma_period).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # 1h volume filter: 1.5x 24-period average (1 day)
    vol_ma_period = 24
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_period, vol_ma_period, sma_period) + 10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper_4h_aligned[i]) or np.isnan(donch_lower_4h_aligned[i]) or 
            np.isnan(donch_middle_4h_aligned[i]) or np.isnan(sma_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                pass  # Hold position outside session
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below 4h Donchian middle or volume drops below average
            if close[i] < donch_middle_4h_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: Price above 4h Donchian middle or volume drops below average
            if close[i] > donch_middle_4h_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: Price above 4h Donchian upper with volume surge and price > 1d SMA50
            if (close[i] > donch_upper_4h_aligned[i] and vol_surge[i] and 
                close[i] > sma_1d_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: Price below 4h Donchian lower with volume surge and price < 1d SMA50
            elif (close[i] < donch_lower_4h_aligned[i] and vol_surge[i] and 
                  close[i] < sma_1d_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals