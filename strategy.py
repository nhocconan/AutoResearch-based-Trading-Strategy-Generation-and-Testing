#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_With_Volume_And_Trend_Filter
Hypothesis: Price breaks above/below Donchian(20) channel with volume confirmation and EMA50 trend filter.
Designed to capture breakout moves while filtering false signals in sideways markets.
Target: 25-35 trades/year to minimize fee drag while capturing institutional breakout moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channel
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: >2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(ema_50[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        ema50 = ema_50[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume spike in uptrend
            if (price > upper and          # breakout above
                vol_spike and              # volume confirmation
                price > ema50):            # uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume spike in downtrend
            elif (price < lower and        # breakout below
                  vol_spike and            # volume confirmation
                  price < ema50):          # downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses back below lower band or trend reverses
            if price < lower or price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses back above upper band or trend reverses
            if price > upper or price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_With_Volume_And_Trend_Filter"
timeframe = "4h"
leverage = 1.0