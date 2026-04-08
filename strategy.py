#!/usr/bin/env python3
# 12h_price_channel_breakout_v1
# Hypothesis: Uses 12h Donchian breakout with volume confirmation and weekly trend filter.
# Long when price breaks above 20-period Donchian high + volume > 1.5x average + weekly close above 50-week SMA.
# Short when price breaks below 20-period Donchian low + volume > 1.5x average + weekly close below 50-week SMA.
# Exit when price crosses back through the middle of the Donchian channel.
# Designed for low trade frequency (15-30/year) to avoid fee drag in ranging/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_price_channel_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    donch_len = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=donch_len, min_periods=donch_len).max().values
    donch_low = low_series.rolling(window=donch_len, min_periods=donch_len).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume filter: 1.5x 20-period average
    vol_ma_len = 20
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=vol_ma_len, min_periods=vol_ma_len).mean().values
    vol_surge = volume > (1.5 * vol_ma)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    weekly_bull = close_1w > sma50_1w
    weekly_bear = close_1w < sma50_1w
    weekly_bull_aligned = align_htf_to_ltf(prices, df_1w, weekly_bull)
    weekly_bear_aligned = align_htf_to_ltf(prices, df_1w, weekly_bear)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_len, vol_ma_len, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_bull_aligned[i]) or 
            np.isnan(weekly_bear_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below Donchian middle
            if close[i] < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above Donchian middle
            if close[i] > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Break above Donchian high + volume surge + weekly bullish
            if (close[i] > donch_high[i] and 
                vol_surge[i] and 
                weekly_bull_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Break below Donchian low + volume surge + weekly bearish
            elif (close[i] < donch_low[i] and 
                  vol_surge[i] and 
                  weekly_bear_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals