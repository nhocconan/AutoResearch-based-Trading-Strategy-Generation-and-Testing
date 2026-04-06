#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 1d EMA(50), volume > 1.5x 20-period avg
# Enter short when: price breaks below Donchian(20) low, price < 1d EMA(50), volume > 1.5x 20-period avg
# Exit when: price reverses to opposite Donchian level (10-period) or opposite extreme RSI
# Uses daily trend to filter counter-trend breaks, targeting 50-150 trades over 4 years (12-37/year)

name = "12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
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
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian channels (10-period for exit)
    high_max_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_min_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below 10-period Donchian low OR RSI > 70 (overbought)
            if close[i] < low_min_10[i]:  # or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 10-period Donchian high OR RSI < 30 (oversold)
            if close[i] > high_max_10[i]:  # or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_max[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above 20-period high and above daily EMA
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_min[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakout below 20-period low and below daily EMA
                    signals[i] = -0.25
                    position = -1
    
    return signals