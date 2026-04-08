#!/usr/bin/env python3
# 6h_volume_price_action_v1
# Hypothesis: Volume-price action strategy on 6h timeframe using volume spikes and price action at key levels.
# Long when price makes higher high with volume spike > 2x average and close > open (bullish candle).
# Short when price makes lower low with volume spike > 2x average and close < open (bearish candle).
# Uses no indicators - pure price and volume action to avoid lag and curve-fitting.
# Works in bull markets (momentum continuation) and bear markets (panic selling exhaustion).
# Target: 20-35 trades/year to stay under fee drag limits.

import numpy as np
import pandas as pd

name = "6h_volume_price_action_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Volume average: 20-period for spike detection
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price action signals
    higher_high = high > np.maximum.accumulate(high)  # New high
    lower_low = low < np.minimum.accumulate(low)      # New low
    bullish_candle = close > open_price
    bearish_candle = close < open_price
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if volume data not available
        if np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: reverse signal or price makes new low
            if bearish_candle[i] and volume[i] > 2.0 * avg_volume[i] and lower_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: reverse signal or price makes new high
            if bullish_candle[i] and volume[i] > 2.0 * avg_volume[i] and higher_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume spike condition
            volume_spike = volume[i] > 2.0 * avg_volume[i]
            
            # Long entry: bullish candle with volume spike at new high
            if bullish_candle[i] and volume_spike and higher_high[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish candle with volume spike at new low
            elif bearish_candle[i] and volume_spike and lower_low[i]:
                position = -1
                signals[i] = -0.25
    
    return signals