#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 12-hour EMA trend filter and volume confirmation
# Enters long when price breaks above 20-period Donchian high with volume > 1.5x average and price above 12h EMA100
# Enters short when price breaks below 20-period Donchian low with volume > 1.5x average and price below 12h EMA100
# Exits when price crosses Donchian midline (average of high/low) or reverses against trend
# Designed for 75-200 trades over 4 years (19-50/year) to minimize fee drag while capturing trends

name = "4h_donchian_12h_ema_vol_v1"
timeframe = "4h"
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
    
    # Donchian channels (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 12-hour EMA100 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_100 = pd.Series(close_12h).ewm(span=100, adjust=False).mean().values
    ema_100_aligned = align_htf_to_ltf(prices, df_12h, ema_100)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_100_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian middle OR closes below 12h EMA100
            if close[i] < donchian_mid[i] or close[i] < ema_100_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian middle OR closes above 12h EMA100
            if close[i] > donchian_mid[i] or close[i] > ema_100_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and trend filter
            if close[i] > donchian_high[i] and volume[i] > volume_threshold[i] and close[i] > ema_100_aligned[i]:
                # Long breakout in uptrend
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_low[i] and volume[i] > volume_threshold[i] and close[i] < ema_100_aligned[i]:
                # Short breakdown in downtrend
                signals[i] = -0.25
                position = -1
    
    return signals