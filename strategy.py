#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Ehlers Fisher Transform on 6h with 1d Trend Filter
# - Ehlers Fisher Transform identifies extreme price movements likely to reverse
# - Long when Fisher crosses above -1.5 (end of sell-off)
# - Short when Fisher crosses below +1.5 (end of rally)
# - 1d trend filter ensures we trade with the higher timeframe trend
# - Works in bull/bear markets by avoiding counter-trend trades
# - Target: 15-30 trades/year to minimize fee drag on 6h timeframe

name = "6h_EhlersFisher_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Ehlers Fisher Transform on close prices
    # Fisher Transform formula: 0.5 * ln((1+X)/(1-X)) where X is normalized price
    # Normalize price to [-1, 1] range over lookback period
    lookback = 10
    highest_high = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(close).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    # Normalize price to [-1, 1]
    normalized_price = 2 * (close - lowest_low) / price_range - 1
    
    # Limit normalized price to [-0.999, 0.999] to avoid infinity in Fisher Transform
    normalized_price = np.clip(normalized_price, -0.999, 0.999)
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + normalized_price) / (1 - normalized_price))
    
    # Smooth the Fisher transform (optional but common)
    fishersmooth = pd.Series(fisher).ewm(span=3, adjust=False).mean().values
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = lookback  # Need enough data for calculation
    
    for i in range(start_idx, n):
        # Skip if trend data is not available
        if np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 AND 1d uptrend
            long_cross = (fishersmooth[i] > -1.5 and fishersmooth[i-1] <= -1.5)
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            # Short: Fisher crosses below +1.5 AND 1d downtrend
            short_cross = (fishersmooth[i] < 1.5 and fishersmooth[i-1] >= 1.5)
            downtrend = ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]
            
            if long_cross and uptrend:
                signals[i] = 0.25
                position = 1
            elif short_cross and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Fisher crosses below -1.5 (mean reversion complete)
            if fishersmooth[i] < -1.5 and fishersmooth[i-1] >= -1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Fisher crosses above +1.5 (mean reversion complete)
            if fishersmooth[i] > 1.5 and fishersmooth[i-1] <= 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals