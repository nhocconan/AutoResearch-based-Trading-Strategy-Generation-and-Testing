#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(20) trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 12h EMA(20), volume > 1.5x avg volume
# Enter short when: price breaks below Donchian(20) low, price < 12h EMA(20), volume > 1.5x avg volume
# Exit when: price crosses back through Donchian(20) median OR opposite breakout occurs
# Uses 12h EMA for trend alignment to avoid counter-trend trades, targeting 80-150 trades over 4 years

name = "4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # 12h EMA(20) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_20 = pd.Series(close_12h).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian mid OR opposite breakout
            if close[i] < donchian_mid[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian mid OR opposite breakout
            if close[i] > donchian_mid[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with trend and volume filters
            if volume[i] > volume_threshold[i]:
                # Long breakout: price above Donchian high and above 12h EMA
                if close[i] > donchian_high[i] and close[i] > ema_20_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below Donchian low and below 12h EMA
                elif close[i] < donchian_low[i] and close[i] < ema_20_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals