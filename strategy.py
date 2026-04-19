#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly EMA100 trend filter and volume spike confirmation.
# Long when: Price breaks above 20-day high, weekly EMA100 upward, volume > 2x 20-day average
# Short when: Price breaks below 20-day low, weekly EMA100 downward, volume > 2x 20-day average
# Exit when: Price crosses back through 20-day moving average
# Donchian provides trend-following structure, weekly EMA100 filters long-term trend, volume confirms breakout strength.
# Target: 10-20 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "1d_Donchian20_EMA100_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA100 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA100 for trend filter
    ema100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align weekly EMA100 to daily timeframe
    ema100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema100_1w)
    
    # 20-period Donchian channels (daily)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-day moving average for exit
    ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 120  # Wait for EMA100 calculation and warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema100_1w_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(ma_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema100 = ema100_1w_aligned[i]
        upper = high_20[i]
        lower = low_20[i]
        ma = ma_20[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above 20-day high, weekly EMA100 upward, volume spike
            if (price > upper and ema100 > ema100_1w_aligned[i-1] and vol > 2.0 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below 20-day low, weekly EMA100 downward, volume spike
            elif (price < lower and ema100 < ema100_1w_aligned[i-1] and vol > 2.0 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below 20-day MA
            if price < ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above 20-day MA
            if price > ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals