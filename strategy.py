#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d trend filter
# Long when: price breaks above Donchian(20) high, volume > 1.5x 20-period avg, price > 1d EMA(200)
# Short when: price breaks below Donchian(20) low, volume > 1.5x 20-period avg, price < 1d EMA(200)
# Exit when: price crosses Donchian(10) midline (10-period Donchian midpoint) or opposite breakout occurs
# Uses daily EMA(200) filter to align with higher timeframe trend, targeting 100-180 trades over 4 years

name = "4h_donchian20_vol_1dema200_trend_v1"
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
    
    # Donchian channels (20-period for entry, 10-period for exit)
    high_roll_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    high_roll_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_roll_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Donchian midline (10-period midpoint)
    donchian_mid = (high_roll_10 + low_roll_10) / 2
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    # 1d EMA(200) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll_20[i]) or np.isnan(low_roll_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_threshold[i]) or 
            np.isnan(ema_200_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian midline OR opposite breakout with volume
            if close[i] < donchian_mid[i] or (close[i] < low_roll_20[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian midline OR opposite breakout with volume
            if close[i] > donchian_mid[i] or (close[i] > high_roll_20[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume confirmation and trend filter
            if volume[i] > volume_threshold[i]:
                # Long breakout: price above Donchian high AND above daily EMA(200)
                if close[i] > high_roll_20[i] and close[i] > ema_200_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below Donchian low AND below daily EMA(200)
                elif close[i] < low_roll_20[i] and close[i] < ema_200_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals