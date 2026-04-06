#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation
# Long when: price breaks above Donchian high(20) AND price > 1d EMA(50) AND volume > 1.5x avg
# Short when: price breaks below Donchian low(20) AND price < 1d EMA(50) AND volume > 1.5x avg
# Exit when: price crosses back through Donchian midpoint OR volume drops below average
# Uses 4h timeframe to balance trade frequency and signal quality. Targets 75-200 total trades over 4 years.
# Works in bull markets (breakouts catch trends) and bear markets (short breakdowns catch declines)

name = "4h_donchian20_1d_ema_vol_v2"
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
    
    # Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    # Calculate EMA on daily close
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean()
    ema_50 = ema_50.values
    
    # Align daily EMA to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Donchian midpoint OR volume drops below average
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or volume[i] <= volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or volume[i] <= volume_ma.values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Donchian breakout + EMA trend filter + volume confirmation
            # Long: price breaks above Donchian high AND price > 1d EMA(50) AND volume confirmation
            if (close[i] > donchian_high[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < 1d EMA(50) AND volume confirmation
            elif (close[i] < donchian_low[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals