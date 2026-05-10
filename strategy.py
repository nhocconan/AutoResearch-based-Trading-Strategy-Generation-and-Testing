#!/usr/bin/env python3
# 6h_Aroon_Trend_Filter_With_Volume_Spike
# Hypothesis: Use Aroon oscillator (25-period) to detect strong trends, enter on pullbacks confirmed by volume spikes.
# AroonUp > 70 and AroonDown < 30 indicates strong uptrend; reverse for downtrend.
# Enter long when price pulls back to EMA(20) during strong uptrend with volume > 1.5x average volume.
# Enter short when price pulls back to EMA(20) during strong downtrend with volume > 1.5x average volume.
# Exit when Aroon trend weakens (AroonUp < 50 or AroonDown < 50) or price crosses EMA(20) in opposite direction.
# Designed to work in both bull and bear markets by following strong trends with volume confirmation.
# Targets ~20-30 trades/year to minimize fee drag.

name = "6h_Aroon_Trend_Filter_With_Volume_Spike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Aroon oscillator (25-period) for trend strength
    period = 25
    aroon_up = np.zeros(n)
    aroon_down = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.argmax(high[i-period+1:i+1])  # 0 to period-1
        lowest_low = np.argmin(low[i-period+1:i+1])    # 0 to period-1
        aroon_up[i] = ((period - 1 - highest_high) / (period - 1)) * 100
        aroon_down[i] = ((period - 1 - lowest_low) / (period - 1)) * 100
    
    # EMA(20) for dynamic support/resistance
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike detector: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, period)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or 
            np.isnan(ema20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: strong uptrend (AroonUp > 70, AroonDown < 30) + pullback to EMA20 + volume spike
            if (aroon_up[i] > 70 and aroon_down[i] < 30 and
                low[i] <= ema20[i] * 1.005 and  # within 0.5% of EMA (pullback)
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend (AroonDown > 70, AroonUp < 30) + pullback to EMA20 + volume spike
            elif (aroon_down[i] > 70 and aroonUp[i] < 30 and
                  high[i] >= ema20[i] * 0.995 and  # within 0.5% of EMA (pullback)
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: trend weakens or price crosses below EMA20
            if (aroon_up[i] < 50 or aroon_down[i] > 50 or
                close[i] < ema20[i] * 0.995):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend weakens or price crosses above EMA20
            if (aroon_down[i] < 50 or aroon_up[i] > 50 or
                close[i] > ema20[i] * 1.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals