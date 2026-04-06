#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-period) AND price > 4h EMA(50) AND volume > 2x 20-period average
# Short when price breaks below Donchian lower (20-period) AND price < 4h EMA(50) AND volume > 2x 20-period average
# Exit when price crosses Donchian midline (10-period average of upper/lower)
# Uses 1h timeframe to capture timely entries, 4h EMA for trend filter, Donchian for breakout signals
# Target: 60-150 total trades over 4 years (15-37/year) for optimal 1h performance
# Session filter: 08-20 UTC to reduce noise trades

name = "1h_donchian20_4h_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # 4-hour EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    fourh_close = df_4h['close'].values
    fourh_close_series = pd.Series(fourh_close)
    fourh_ema = fourh_close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 4h EMA to 1h timeframe
    fourh_ema_aligned = align_htf_to_ltf(prices, df_4h, fourh_ema)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(fourh_ema_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Apply session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above Donchian upper AND price > 4h EMA AND volume confirmation
            if (close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1] and 
                close[i] > fourh_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian lower AND price < 4h EMA AND volume confirmation
            elif (close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1] and 
                  close[i] < fourh_ema_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals