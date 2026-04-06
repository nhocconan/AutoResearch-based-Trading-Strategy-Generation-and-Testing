#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA trend filter and volume confirmation
# Enter long when: price breaks above 20-period high, price > 1d EMA50, volume > 1.5x 20-period average
# Enter short when: price breaks below 20-period low, price < 1d EMA50, volume > 1.5x 20-period average
# Exit when: price crosses back through the Donchian midpoint or opposite breakout occurs
# Uses 1d trend filter to avoid counter-trend trades, volume to confirm breakout strength
# Target: 100-200 trades over 4 years by requiring multiple confluence factors

name = "4h_donchian20_1d_ema_vol_v2"
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
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian midpoint OR opposite breakout
            if close[i] < donchian_mid[i] or (low[i] < low_roll[i] and close[i] < open_price if 'open_price' in locals() else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian midpoint OR opposite breakout
            if close[i] > donchian_mid[i] or (high[i] > high_roll[i] and close[i] > open_price if 'open_price' in locals() else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend and volume confirmation
            if volume[i] > volume_threshold[i]:
                # Long breakout: price above upper band with uptrend
                if high[i] > high_roll[i] and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below lower band with downtrend
                elif low[i] < low_roll[i] and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals