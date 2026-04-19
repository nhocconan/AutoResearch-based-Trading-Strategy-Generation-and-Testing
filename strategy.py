#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1-day EMA50 trend filter + volume spike confirmation.
# Long when: price breaks above Donchian high(20), price > EMA50(1d), volume > 2x 20-period average
# Short when: price breaks below Donchian low(20), price < EMA50(1d), volume > 2x 20-period average
# Exit when price crosses back through the opposite Donchian boundary (long exit below low(20), short exit above high(20))
# Uses price channel breakouts for trend following, EMA50 for trend filter, volume for conviction.
# Target: 20-40 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "4h_Donchian20_EMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels on 4h data (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        ema50 = ema50_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high, price > EMA50, volume spike
            if (price > upper and price > ema50 and vol > 2.0 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low, price < EMA50, volume spike
            elif (price < lower and price < ema50 and vol > 2.0 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below Donchian low
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above Donchian high
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals