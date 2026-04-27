#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA trend filter and volume spike.
# Williams %R identifies overbought/oversold conditions.
# Readings below -80 = oversold (buy signal), above -20 = overbought (sell signal).
# Strategy: In uptrends (price > 1d EMA50), buy when Williams %R crosses above -80 from below.
# In downtrends (price < 1d EMA50), sell when Williams %R crosses below -20 from above.
# Volume spike (>1.5x 20-period average) confirms institutional participation.
# Designed for ~15-30 trades/year per symbol (~60-120 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    willr = -100 * (highest_high - close) / hl_range  # Values between -100 and 0
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(willr[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Uptrend: price above 1d EMA50
        if close[i] > ema50_1d_aligned[i]:
            # Buy signal: Williams %R crosses above -80 from below (oversold bounce)
            if willr[i] > -80 and willr[i-1] <= -80:
                if volume_filter[i]:
                    signals[i] = 0.25
                    position = 1
            # Exit long: Williams %R rises above -20 (overbought)
            elif position == 1 and willr[i] >= -20:
                signals[i] = 0.0
                position = 0
        # Downtrend: price below 1d EMA50
        elif close[i] < ema50_1d_aligned[i]:
            # Sell signal: Williams %R crosses below -20 from above (overbought reversal)
            if willr[i] < -20 and willr[i-1] >= -20:
                if volume_filter[i]:
                    signals[i] = -0.25
                    position = -1
            # Exit short: Williams %R falls below -80 (oversold)
            elif position == -1 and willr[i] <= -80:
                signals[i] = 0.0
                position = 0
        # Neutral: price near EMA50 - hold flat
        else:
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsR_1dEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0