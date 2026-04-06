#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-period) AND price > 12h EMA(60) AND volume > 1.8x 20-period average
# Short when price breaks below Donchian lower (20-period) AND price < 12h EMA(60) AND volume > 1.8x 20-period average
# Exit when price crosses Donchian midline (10-period average of upper/lower)
# Uses 4h timeframe for balanced trade frequency, 12h EMA for trend filter, Donchian for breakout signals
# Target: 75-200 total trades over 4 years (19-50/year) for optimal 4h performance

name = "4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # 12-hour EMA(60) trend filter
    df_12h = get_htf_data(prices, '12h')
    twelve_hour_close = df_12h['close'].values
    
    # Calculate 60-period EMA on 12h close
    twelve_hour_close_series = pd.Series(twelve_hour_close)
    twelve_hour_ema = twelve_hour_close_series.ewm(span=60, min_periods=60, adjust=False).mean().values
    
    # Align 12h EMA to 4h timeframe
    twelve_hour_ema_aligned = align_htf_to_ltf(prices, df_12h, twelve_hour_ema)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.8 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(twelve_hour_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above Donchian upper AND price > 12h EMA AND volume confirmation
            if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                close[i] > twelve_hour_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < 12h EMA AND volume confirmation
            elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                  close[i] < twelve_hour_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals