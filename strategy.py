#!/usr/bin/env python3
"""
4h_Donchian_Breakout_20_SMA50_Trend_Volume
Hypothesis: Breakouts above/below 20-period Donchian channel on 4h timeframe, 
confirmed by 50-period SMA trend and volume spike, capture significant moves 
in both bull and bear markets. The Donchian channel acts as dynamic support/resistance,
while SMA50 filters for trend alignment and volume ensures conviction. 
Targets 20-50 trades/year on 4h to minimize fee drag while capturing strong directional moves.
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
    
    # Get 4h data for Donchian calculation (though we're on 4h, we use it for consistency)
    # Actually, we're on 4h timeframe, so we can calculate directly
    # But we'll still use mtf_data for consistency and potential future HTF use
    
    # Calculate Donchian channel (20-period high/low)
    # Using pandas rolling for clarity
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Donchian upper = 20-period high, lower = 20-period low
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 50-period SMA for trend filter
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with uptrend and volume spike
            if close[i] > donchian_high[i] and close[i] > sma50[i] and vol_spike[i]:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with downtrend and volume spike
            elif close[i] < donchian_low[i] and close[i] < sma50[i] and vol_spike[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below Donchian low or trend turns down
            if close[i] < donchian_low[i] or close[i] < sma50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above Donchian high or trend turns up
            if close[i] > donchian_high[i] or close[i] > sma50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_20_SMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0