#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Channels with Volume Confirmation and Trend Filter
# Hypothesis: Price breaking above/below daily Donchian channels (20-period high/low) 
# with volume confirmation and daily trend filter (price vs daily 20 EMA) captures 
# momentum moves while avoiding whipsaws. In bull markets: buy breakouts above daily high. 
# In bear markets: sell breakouts below daily low. Daily timeframe provides sufficient 
# signal quality while 4h execution allows timely entry. Volume confirms institutional 
# participation. Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_channels_volume_trend_v1"
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
    
    # Get daily data for channels and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period high/low)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    daily_high_series = pd.Series(daily_high)
    daily_low_series = pd.Series(daily_low)
    donchian_high = daily_high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = daily_low_series.rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only completed daily bars (avoid look-ahead)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    
    # Handle first element
    if len(donchian_high) > 1:
        donchian_high[0] = donchian_high[1]
        donchian_low[0] = donchian_low[1]
    else:
        donchian_high[0] = 0
        donchian_low[0] = 0
    
    # Daily trend filter: price vs 20 EMA
    daily_close_series = pd.Series(daily_close)
    daily_ema_20 = daily_close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    daily_ema_20 = np.roll(daily_ema_20, 1)
    if len(daily_ema_20) > 1:
        daily_ema_20[0] = daily_ema_20[1]
    else:
        daily_ema_20[0] = 0
    
    # Align daily data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_daily, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_daily, donchian_low)
    daily_ema_20_aligned = align_htf_to_ltf(prices, df_daily, daily_ema_20)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(daily_ema_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below daily Donchian low or trend fails
            if close[i] < donchian_low_aligned[i] or close[i] < daily_ema_20_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above daily Donchian high or trend fails
            if close[i] > donchian_high_aligned[i] or close[i] > daily_ema_20_aligned[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: breakout above daily Donchian high with volume and trend filter
            if (high[i] > donchian_high_aligned[i] and close[i] > donchian_high_aligned[i] and
                close[i] > daily_ema_20_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below daily Donchian low with volume and trend filter
            elif (low[i] < donchian_low_aligned[i] and close[i] < donchian_low_aligned[i] and
                  close[i] < daily_ema_20_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals