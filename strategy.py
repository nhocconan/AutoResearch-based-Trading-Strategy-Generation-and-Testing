#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Supertrend_1dTrend_HTFVolumeFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = (close_1d > ema50_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 1d volume filter: volume > 1.5 * 20-period average
    vol_ma20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_filter_1d = df_1d['volume'].values > (vol_ma20_1d * 1.5)
    vol_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_filter_1d)
    
    # Supertrend on 6h: ATR(10) * 3
    atr_period = 10
    atr_mult = 3.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic upper and lower bands
    hl2 = (high + low) / 2
    upperband = hl2 + (atr_mult * atr)
    lowerband = hl2 - (atr_mult * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > upperband[i-1]:
            direction[i] = 1
        elif close[i] < lowerband[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
            if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            if direction[i] == -1 and upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
        
        supertrend[i] = upperband[i] if direction[i] == -1 else lowerband[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 100)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(vol_filter_1d_aligned[i]) or 
            np.isnan(supertrend[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Supertrend uptrend, 1d uptrend, and volume filter
            long_cond = (direction[i] == 1 and trend_1d_aligned[i] > 0.5 and vol_filter_1d_aligned[i])
            
            # Short entry: Supertrend downtrend, 1d downtrend, and volume filter
            short_cond = (direction[i] == -1 and trend_1d_aligned[i] < 0.5 and vol_filter_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Supertrend flips to downtrend
            if direction[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Supertrend flips to uptrend
            if direction[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Supertrend on 6h timeframe with 1d EMA50 trend filter and volume filter.
# Enters long when Supertrend is uptrend, 1d close > EMA50, and 1d volume > 1.5x 20-period average.
# Enters short when Supertrend is downtrend, 1d close < EMA50, and 1d volume > 1.5x 20-period average.
# Exits when Supertrend flips direction.
# Uses Supertrend(ATR=10, multiplier=3) for trend following, combined with higher timeframe
# trend and volume confirmation to filter false signals. Works in trending markets (both bull and bear)
# by only taking trades in the direction of the 1d trend with volume confirmation.
# Targets 20-40 trades/year on 6h timeframe to avoid overtrading. Uses discrete sizing (0.25) to minimize churn.