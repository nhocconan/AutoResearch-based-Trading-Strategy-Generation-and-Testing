#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ehlers Fisher Transform with 1w Trend and Volume Filter
# - Fisher Transform (length=10) identifies turning points in price
# - Long when Fisher crosses above -1.5, short when crosses below +1.5
# - 1w trend filter (EMA34) ensures alignment with higher timeframe direction
# - Volume filter avoids low-liquidity false signals
# - Works in bull/bear by using 1w trend to avoid counter-trend trades
# - Target: 20-40 trades/year to minimize fee drag on 6h timeframe

name = "6h_FisherTransform_1wTrend_Volume"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Ehlers Fisher Transform (length=10)
    # Price = (High + Low) / 2
    price = (high + low) / 2.0
    
    # Normalize price to [-1, 1] range over lookback period
    def normalize_series(series, length):
        highest = pd.Series(series).rolling(window=length, min_periods=length).max().values
        lowest = pd.Series(series).rolling(window=length, min_periods=length).min().values
        # Avoid division by zero
        range_val = highest - lowest
        range_val = np.where(range_val == 0, 1, range_val)
        normalized = 2 * ((series - lowest) / range_val) - 1
        # Clamp to [-0.999, 0.999] to prevent math domain errors
        normalized = np.clip(normalized, -0.999, 0.999)
        return normalized
    
    price_norm = normalize_series(price, 10)
    
    # Fisher Transform: 0.5 * ln((1 + x) / (1 - x))
    fish = 0.5 * np.log((1 + price_norm) / (1 - price_norm))
    
    # Volume filter: current volume > 1.5x 30-period average
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Fisher calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(fish[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 + 1w uptrend + volume filter
            long_cross = (fish[i] > -1.5 and fish[i-1] <= -1.5)
            long_cond = long_cross and (ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]) and volume_filter[i]
            
            # Short: Fisher crosses below +1.5 + 1w downtrend + volume filter
            short_cross = (fish[i] < 1.5 and fish[i-1] >= 1.5)
            short_cond = short_cross and (ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Fisher crosses below -1.5 (reversal signal)
            if fish[i] < -1.5 and fish[i-1] >= -1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Fisher crosses above +1.5 (reversal signal)
            if fish[i] > 1.5 and fish[i-1] <= 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals