#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) with price close outside all three lines
# and volume confirmation. The Alligator identifies trending vs ranging markets.
# When jaws (13-period SMMA) are below teeth (8-period SMMA) and lips (5-period SMMA),
# it indicates a downtrend; reverse for uptrend. We enter when price closes outside
# the Alligator's mouth (beyond lips) in the direction of the trend, with volume > 1.5x 20-period EMA.
# Exit when price re-enters the Alligator's mouth (between jaws and lips).
# Designed to catch strong trends while avoiding chop, suitable for both bull and bear markets.
name = "4h_WilliamsAlligator_Trend_Volume"
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
    
    # Williams Alligator: SMMA (Smoothed Moving Average) - using EMA as approximation
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Uptrend: lips > teeth > jaw AND price closes above lips (bullish alignment)
            # Downtrend: jaw > teeth > lips AND price closes below jaws (bearish alignment)
            if (lips[i] > teeth[i] > jaw[i] and price > lips[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            elif (jaw[i] > teeth[i] > lips[i] and price < jaw[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price re-enters Alligator's mouth (below lips)
            if price < lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price re-enters Alligator's mouth (above jaws)
            if price > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals