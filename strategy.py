#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 12h trend filter
# Enter long when: price breaks above Donchian(20) high, volume > 1.5x average, price > 12h EMA(50)
# Enter short when: price breaks below Donchian(20) low, volume > 1.5x average, price < 12h EMA(50)
# Exit when price reverses back to Donchian midpoint or opposite breakout occurs
# Uses 12h EMA to filter trades against higher timeframe trend, targeting 80-150 trades over 4 years
# Focus on volatility expansion breakouts with volume confirmation to avoid false signals

name = "4h_donchian20_12h_ema_vol_v1"
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
    
    # Donchian Channel (20-period)
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_rolling.values
    donchian_low = low_rolling.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to Donchian midpoint OR breaks below Donchian low
            if close[i] <= donchian_mid[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to Donchian midpoint OR breaks above Donchian high
            if close[i] >= donchian_mid[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume and trend confirmation
            if volume[i] > volume_threshold[i]:
                # Long breakout: price breaks above Donchian high with uptrend filter
                if close[i] > donchian_high[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price breaks below Donchian low with downtrend filter
                elif close[i] < donchian_low[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals