# 4h_Donchian_Breakout_With_Volume_Confirmation
# Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) upper band with above-average volume, and short when price breaks below Donchian(20) lower band with above-average volume.
# Exit when price touches the opposite band (mean reversion within the channel).
# Uses volume confirmation to filter false breakouts and ATR-based position sizing (0.25).
# Works in both bull and bear markets: breakouts capture trends, while mean reversion exit prevents large drawdowns during sideways periods.
# Target: ~25-40 trades/year per symbol to stay under fee drag threshold.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Calculate average volume (20-period) for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Volume ratio: current volume / average volume
    volume_ratio = np.zeros(n)
    volume_ratio[20:] = volume[20:] / avg_volume[20:]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above upper band with volume confirmation
            if close[i] > upper[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band with volume confirmation
            elif close[i] < lower[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price touches or goes below lower band (mean reversion)
            if close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches or goes above upper band (mean reversion)
            if close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0