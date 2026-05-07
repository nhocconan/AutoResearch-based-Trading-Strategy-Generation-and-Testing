#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (13,8,5 SMAs) with 1d trend filter and volume confirmation.
# Long when Alligator jaws (13 SMA) > teeth (8 SMA) > lips (5 SMA) AND 1d EMA34 rising AND volume > 1.5x 20-period average.
# Short when jaws < teeth < lips AND 1d EMA34 falling AND volume > 1.5x 20-period average.
# Exit when Alligator lines re-cross (jaws < teeth or jaws > lips depending on position).
# The Alligator identifies trending vs ranging markets. Teeth and lips convergence indicates trend exhaustion.
# Combined with 1d trend filter ensures we trade with higher timeframe momentum.
# Volume confirmation filters out false breakouts. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (SMAs of median price)
    median_price = (high + low) / 2
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values  # 5-period SMA
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values  # 8-period SMA
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20)  # Sufficient warmup for jaws and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaws[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: jaws > teeth > lips (bullish alignment), 1d EMA34 rising, volume filter
            long_cond = (jaws[i] > teeth[i]) and (teeth[i] > lips[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: jaws < teeth < lips (bearish alignment), 1d EMA34 falling, volume filter
            short_cond = (jaws[i] < teeth[i]) and (teeth[i] < lips[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator lines re-cross (jaws < teeth or teeth < lips)
            if jaws[i] < teeth[i] or teeth[i] < lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator lines re-cross (jaws > teeth or teeth > lips)
            if jaws[i] > teeth[i] or teeth[i] > lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals