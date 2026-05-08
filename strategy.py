#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band breakout with 12h trend filter and volume confirmation.
# Long when price breaks above BB upper AND 12h close > 12h EMA50 AND volume > 1.5x 20-period avg.
# Short when price breaks below BB lower AND 12h close < 12h EMA50 AND volume > 1.5x 20-period avg.
# Exit when price crosses back below/above BB middle.
# Bollinger Bands capture volatility expansion, EMA50 filters trend direction, volume confirms strength.
# Target: 80-150 total trades over 4 years (20-38/year) for low fee drift.

name = "4h_BB_12hEMA_Volume"
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
    
    # 4h Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_middle = sma20
    
    # 4h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above BB upper, 12h close > EMA50, volume spike
            long_cond = (close[i] > bb_upper[i]) and (close_12h[i] > ema50_12h[i]) and volume_filter[i]
            # Short conditions: break below BB lower, 12h close < EMA50, volume spike
            short_cond = (close[i] < bb_lower[i]) and (close_12h[i] < ema50_12h[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below BB middle
            if close[i] < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above BB middle
            if close[i] > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals