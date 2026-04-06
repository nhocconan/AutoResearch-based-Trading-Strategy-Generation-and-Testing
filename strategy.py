#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h Donchian breakout direction and volume confirmation
# Long when price breaks above 4h Donchian upper (20-period) AND price > 4h EMA(50) AND volume > 2x 20-period average
# Short when price breaks below 4h Donchian lower (20-period) AND price < 4h EMA(50) AND volume > 2x 20-period average
# Exit when price crosses 4h Donchian midline
# Use 1h only for entry timing, 4h for signal direction
# Add session filter (08-20 UTC) to reduce noise trades
# Target: 60-150 total trades over 4 years = 15-37/year for 1h

name = "1h_4h_donchian20_4h_ema_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute hour for session filter
    hours = prices.index.hour
    
    # 4h Donchian Channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max()
    lowest_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min()
    donchian_upper_4h = highest_high_4h.values
    donchian_lower_4h = lowest_low_4h.values
    donchian_mid_4h = (donchian_upper_4h + donchian_lower_4h) / 2
    
    # Align 4h Donchian to 1h timeframe
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # 4h EMA(50) trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Skip if required data not available
        if np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or \
           np.isnan(ema_4h_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses 4h Donchian midline
        if position == 1:  # long position
            if close[i] < donchian_mid_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] > donchian_mid_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above 4h Donchian upper AND price > 4h EMA AND volume confirmation
            if (close[i] > donchian_upper_4h_aligned[i] and close[i-1] <= donchian_upper_4h_aligned[i-1] and 
                close[i] > ema_4h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower AND price < 4h EMA AND volume confirmation
            elif (close[i] < donchian_lower_4h_aligned[i] and close[i-1] >= donchian_lower_4h_aligned[i-1] and 
                  close[i] < ema_4h_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals