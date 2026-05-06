# 1h strategy: 4h Donchian breakout with volume confirmation and session filter
# Uses 4h Donchian channels for directional bias and 1h for entry timing
# Designed for 15-35 trades/year with 0.20 position sizing to manage drawdown
# Session filter (08-20 UTC) reduces noise trades
# Works in bull via breakouts above resistance, bear via breakdowns below support

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4hDonchian20_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian Channel (20-period high/low)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 20-period high and low for Donchian channels
    high_20 = df_4h['high'].rolling(window=20, min_periods=20).max().values
    low_20 = df_4h['low'].rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    upper_donchian = align_htf_to_ltf(prices, df_4h, high_20)
    lower_donchian = align_htf_to_ltf(prices, df_4h, low_20)
    
    # Volume confirmation: >1.5x 20-period average (1h volume)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper Donchian with volume confirmation
            if close[i] > upper_donchian[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short breakout: price breaks below lower Donchian with volume confirmation
            elif close[i] < lower_donchian[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (support break)
            if close[i] < lower_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above upper Donchian (resistance break)
            if close[i] > upper_donchian[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals