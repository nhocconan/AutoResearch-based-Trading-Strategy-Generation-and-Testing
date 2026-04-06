#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d EMA filter and volume confirmation
# Enter long when: price > Alligator Jaw (13-period smoothed median) AND price > 1d EMA(50) AND volume > 1.5x avg
# Enter short when: price < Alligator Jaw AND price < 1d EMA(50) AND volume > 1.5x avg
# Exit when price crosses back below/above Alligator Jaw
# Uses Williams Alligator to identify trending regimes and EMA for higher timeframe trend filter
# Targets 100-200 trades over 4 years (25-50/year) with strong trend following in both bull and bear markets

name = "6h_williams_alligator_1dema_vol_v1"
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
    volume = prices['volume'].values
    
    # Williams Alligator components (13, 8, 5 period SMMedians)
    # Jaw: 13-period smoothed median (SMMA)
    close_series = pd.Series(close)
    jaw_raw = close_series.rolling(window=13, min_periods=13).apply(lambda x: np.median(x), raw=False)
    jaw = jaw_raw.ewm(alpha=1/13, adjust=False, ignore_na=False).mean().values
    
    # Teeth: 8-period smoothed median
    teeth_raw = close_series.rolling(window=8, min_periods=8).apply(lambda x: np.median(x), raw=False)
    teeth = teeth_raw.ewm(alpha=1/8, adjust=False, ignore_na=False).mean().values
    
    # Lips: 5-period smoothed median
    lips_raw = close_series.rolling(window=5, min_periods=5).apply(lambda x: np.median(x), raw=False)
    lips = lips_raw.ewm(alpha=1/5, adjust=False, ignore_na=False).mean().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Wait for Alligator to stabilize
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Alligator Jaw OR price < 1d EMA(50)
            if close[i] < jaw[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Alligator Jaw OR price > 1d EMA(50)
            if close[i] > jaw[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price vs Jaw + 1d EMA filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > jaw[i] and close[i] > ema_50_aligned[i]:
                    # Price above Alligator Jaw and above daily EMA - bullish trend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < jaw[i] and close[i] < ema_50_aligned[i]:
                    # Price below Alligator Jaw and below daily EMA - bearish trend
                    signals[i] = -0.25
                    position = -1
    
    return signals