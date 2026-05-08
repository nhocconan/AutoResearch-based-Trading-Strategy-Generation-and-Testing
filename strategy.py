#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action at 12h Bollinger Bands with volume confirmation.
# Long when price touches lower Bollinger Band (20,2) AND closes above it AND volume > 1.5x 20-period average.
# Short when price touches upper Bollinger Band (20,2) AND closes below it AND volume > 1.5x 20-period average.
# Exit when price crosses back to the 12h EMA20 (mean reversion target).
# Bollinger Bands capture volatility expansion/contraction; mean reversion works in ranging markets.
# EMA20 filter avoids trading against the intermediate trend. Volume confirms institutional participation.
# Target: 60-100 total trades over 4 years (15-25/year) to avoid fee drag.

name = "4h_BollingerBands_12hEMA20_Volume"
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
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 12h data for Bollinger Bands and EMA20
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Bollinger Bands (20,2) on 12h close
    close_12h = df_12h['close'].values
    bb_period = 20
    bb_std = 2
    bb_ma = pd.Series(close_12h).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close_12h).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + (bb_std * bb_std_dev)
    bb_lower = bb_ma - (bb_std * bb_std_dev)
    
    # EMA20 on 12h close for mean reversion target
    ema_20 = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for Bollinger Bands and EMA20
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: touch lower BB, close above it, volume filter
            long_cond = (low[i] <= bb_lower_aligned[i]) and (close[i] > bb_lower_aligned[i]) and volume_filter[i]
            # Short conditions: touch upper BB, close below it, volume filter
            short_cond = (high[i] >= bb_upper_aligned[i]) and (close[i] < bb_upper_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross back to EMA20 (mean reversion target)
            if close[i] >= ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross back to EMA20 (mean reversion target)
            if close[i] <= ema_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals