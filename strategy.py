#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h/1d trend filters and volume confirmation
# Long when price breaks above 1h Donchian high + 4h MA up + 1d MA up + volume > 2x average
# Short when price breaks below 1h Donchian low + 4h MA down + 1d MA down + volume > 2x average
# Exit when price crosses 1h Donchian midpoint or trend reverses
# Uses 4h/1d for trend direction, 1h only for entry timing
# Session filter: 08-20 UTC to avoid low-volume periods
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag

name = "1h_donchian_4h1d_ma_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 4h MA (50-period) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ma_4h = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
    ma_4h_aligned = align_htf_to_ltf(prices, df_4h, ma_4h)
    
    # 1d MA (50-period) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ma_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    ma_1d_aligned = align_htf_to_ltf(prices, df_1d, ma_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available or outside session
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ma_4h_aligned[i]) or np.isnan(ma_1d_aligned[i]) or
            np.isnan(volume_threshold[i]) or not in_session[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Donchian midpoint or trend reverses
        if position == 1:  # long position
            if close[i] <= donch_mid[i] or ma_4h_aligned[i] < ma_4h_aligned[i-1] or ma_1d_aligned[i] < ma_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] >= donch_mid[i] or ma_4h_aligned[i] > ma_4h_aligned[i-1] or ma_1d_aligned[i] > ma_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for breakouts with trend and volume confirmation
            # Bullish breakout: price above Donchian high + both MAs up + volume
            if (close[i] > donch_high[i] and 
                ma_4h_aligned[i] > ma_4h_aligned[i-1] and 
                ma_1d_aligned[i] > ma_1d_aligned[i-1] and
                volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Bearish breakout: price below Donchian low + both MAs down + volume
            elif (close[i] < donch_low[i] and 
                  ma_4h_aligned[i] < ma_4h_aligned[i-1] and 
                  ma_1d_aligned[i] < ma_1d_aligned[i-1] and
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals