#!/usr/bin/env python3
# 4h_price_channel_volume_breakout_v1
# Hypothesis: Price channel breakouts with volume confirmation and trend filter work in both bull and bear markets.
# Long when price breaks above Donchian upper (20) with volume > 1.5x average and 12h EMA(50) > EMA(200).
# Short when price breaks below Donchian lower (20) with volume > 1.5x average and 12h EMA(50) < EMA(200).
# Exit when price returns to Donchian middle or volume drops below average.
# Uses Donchian channels from 4h timeframe, EMA from 12h for trend filter, volume for confirmation.
# Target: 20-50 trades/year with strict entry conditions to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_price_channel_volume_breakout_v1"
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
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Trend: 1 = bullish (EMA50 > EMA200), -1 = bearish (EMA50 < EMA200)
    trend_12h = np.where(ema_50_12h_aligned > ema_200_12h_aligned, 1, -1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(dc_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(dc_middle[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(trend_12h[i])):
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
            # Long entry: Price above upper DC with volume surge and bullish trend
            if (close[i] > dc_upper[i] and vol_surge[i] and trend_12h[i] == 1):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower DC with volume surge and bearish trend
            elif (close[i] < dc_lower[i] and vol_surge[i] and trend_12h[i] == -1):
                position = -1
                signals[i] = -0.25
    
    return signals