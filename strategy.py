#!/usr/bin/env python3
# 4h_price_channel_breakout_volume_v3
# Hypothesis: Price channel breakouts with volume confirmation and daily trend filter.
# Long when price breaks above Donchian upper channel (20) with volume > 1.5x average and daily close > daily EMA50.
# Short when price breaks below Donchian lower channel (20) with volume > 1.5x average and daily close < daily EMA50.
# Exit when price returns to Donchian middle channel or volume drops below average.
# Uses Donchian channels from 4h timeframe, EMA50 from daily timeframe for trend filter.
# Target: 25-50 trades/year with strict entry conditions to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_price_channel_breakout_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20) - calculated on 4h data
    dc_period = 20
    
    # Calculate rolling max/min
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    dc_upper = high_series.rolling(window=dc_period, min_periods=dc_period).max().values
    dc_lower = low_series.rolling(window=dc_period, min_periods=dc_period).min().values
    dc_middle = (dc_upper + dc_lower) / 2
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = pd.Series(close_1d).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(dc_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(dc_middle[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below middle DC or volume drops below average
            if close[i] < dc_middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above middle DC or volume drops below average
            if close[i] > dc_middle[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above upper DC with volume surge and daily uptrend
            if (close[i] > dc_upper[i] and vol_surge[i] and close[i] > ema_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower DC with volume surge and daily downtrend
            elif (close[i] < dc_lower[i] and vol_surge[i] and close[i] < ema_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals