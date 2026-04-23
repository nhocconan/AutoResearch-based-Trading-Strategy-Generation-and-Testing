#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian breakout with 4h EMA50 trend filter and volume spike confirmation.
Long when price breaks above 1h Donchian upper (20) AND 4h EMA50 up AND volume > 1.8x 20-period average.
Short when price breaks below 1h Donchian lower (20) AND 4h EMA50 down AND volume > 1.8x 20-period average.
Exit when price crosses 1h Donchian midpoint (middle band).
Uses discrete position sizing (0.20) to minimize fee churn. Targets 15-35 trades/year per symbol.
1h Donchian provides clear breakout levels with proven efficacy on BTC/ETH pairs.
4h EMA50 offers smooth trend filter for higher timeframe alignment with lower lag than slower MA.
Volume confirmation at 1.8x ensures only significant breakouts are taken, reducing false signals.
Designed to work in both bull (trend following) and bear (counter-trend retracements) markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20)  # Ensure warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND 4h EMA50 up AND volume spike
            if (price > donchian_upper[i] and 
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and 
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian lower AND 4h EMA50 down AND volume spike
            elif (price < donchian_lower[i] and 
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and 
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian midpoint (middle band)
            if position == 1 and price < donchian_middle[i]:
                exit_signal = True
            elif position == -1 and price > donchian_middle[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Donchian20_4hEMA50_VolumeSpike"
timeframe = "1h"
leverage = 1.0