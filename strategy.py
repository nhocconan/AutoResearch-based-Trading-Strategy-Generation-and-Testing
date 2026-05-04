#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Williams Alligator (jaw=13, teeth=8, lips=5) identifies trend via alignment:
#   Jaw (13-period SMMA) > Teeth (8-period SMMA) > Lips (5-period SMMA) = uptrend
#   Reverse = downtrend
#   Intertwined = ranging/choppy
# Only trade when Alligator is aligned (trending) AND price is outside the Alligator's mouth
#   (price > Lips for long, price < Jaw for short) with volume confirmation (>1.3x 20 EMA)
# Use 1d EMA50 as higher timeframe trend filter: only take longs when price > 1d EMA50,
#   shorts when price < 1d EMA50 to avoid counter-trend trades
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 trades over 4 years.
# Strategy works in bull/bear via Alligator's trend identification and 1d EMA50 filter.

name = "6h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator: 3 smoothed moving averages (SMMA)
    # SMMA is similar to EMA but with different smoothing: SMMA(t) = (SMMA(t-1)*(n-1) + close(t))/n
    # We'll approximate with EMA for simplicity and performance
    close_s = pd.Series(close)
    
    # Lips: 5-period SMMA (approx with EMA)
    lips = close_s.ewm(span=5, adjust=False, min_periods=5).mean().values
    # Teeth: 8-period SMMA
    teeth = close_s.ewm(span=8, adjust=False, min_periods=8).mean().values
    # Jaw: 13-period SMMA
    jaw = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        # Alligator alignment: Jaw > Teeth > Lips = uptrend, Jaw < Teeth < Lips = downtrend
        alligator_long = jaw[i] > teeth[i] and teeth[i] > lips[i]
        alligator_short = jaw[i] < teeth[i] and teeth[i] < lips[i]
        
        # Price outside Alligator's mouth: price > Lips for long, price < Jaw for short
        price_above_lips = close[i] > lips[i]
        price_below_jaw = close[i] < jaw[i]
        
        if position == 0:
            # Long conditions: uptrend + price above lips + 1d uptrend + volume
            if (alligator_long and price_above_lips and 
                close[i] > ema50_aligned[i] and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short conditions: downtrend + price below jaw + 1d downtrend + volume
            elif (alligator_short and price_below_jaw and 
                  close[i] < ema50_aligned[i] and volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses alignment OR price re-enters mouth (below teeth) OR 1d trend flips
            if not (alligator_long and price_above_lips and close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses alignment OR price re-enters mouth (above jaw) OR 1d trend flips
            if not (alligator_short and price_below_jaw and close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals