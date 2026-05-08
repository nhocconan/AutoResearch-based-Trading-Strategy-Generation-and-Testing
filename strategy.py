#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation
# Exploits volatility contraction/expansion cycles: enter when price breaks out of
# a low-volatility Bollinger Band squeeze, in the direction of the daily trend.
# Volume spike confirms the breakout. Designed for low trade frequency and robustness
# across bull/bear regimes via trend filter.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_BB_Squeeze_1dTrend_Volume"
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
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Bollinger Bands (20, 2) on 6h data
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    
    # Bollinger Band Width: (upper - lower) / sma20
    bb_width = (upper - lower) / sma20
    # Squeeze condition: BB width below its 50-period mean (low volatility)
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(sma20[i]) or 
            np.isnan(std20[i]) or np.isnan(bb_width[i]) or 
            np.isnan(bb_width_ma[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        is_squeeze = squeeze[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: squeeze breakout up + uptrend + volume spike
            if (close[i] > upper[i] and is_squeeze and 
                close[i] > ema34_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: squeeze breakout down + downtrend + volume spike
            elif (close[i] < lower[i] and is_squeeze and 
                  close[i] < ema34_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Bollinger Bands or trend turns bearish
            if (close[i] < sma20[i] or close[i] < ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Bollinger Bands or trend turns bullish
            if (close[i] > sma20[i] or close[i] > ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals