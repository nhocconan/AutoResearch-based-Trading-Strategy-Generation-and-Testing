#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# - Weekly pivot levels calculated from prior week's OHLC (1d timeframe)
# - Long when price breaks above Donchian(20) high AND price > weekly pivot point AND volume > 1.5x 20-period average
# - Short when price breaks below Donchian(20) low AND price < weekly pivot point AND volume > 1.5x 20-period average
# - Exit when price crosses Donchian(20) midline
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Weekly pivot provides institutional reference point for bias
# - Donchian breakout captures momentum with volume confirmation reducing false signals

name = "6h_1d_weekly_pivot_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Pre-compute 6h Donchian channels (20)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian high/low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Pre-compute 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d weekly pivot points (from prior week's OHLC)
    # Weekly pivot = (Prior week HIGH + LOW + CLOSE) / 3
    weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
    weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1)    # Prior week
    weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1) # Prior week
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Donchian high AND price > weekly pivot AND volume spike
            if (close[i] > donch_high[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Donchian low AND price < weekly pivot AND volume spike
            elif (close[i] < donch_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit condition: price crosses Donchian midline
            exit_long = (position == 1 and close[i] < donch_mid[i])
            exit_short = (position == -1 and close[i] > donch_mid[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals