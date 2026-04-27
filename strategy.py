#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h EMA(50) trend filter with volume spike confirmation.
# Enters long when price is above 12h EMA(50) and volume > 2x 20-period average,
# short when price below 12h EMA(50) and volume > 2x 20-period average.
# Exits when price crosses back below/above EMA or volume drops.
# Designed for ~20-30 trades/year by requiring both trend alignment and volume spike.
# Works in bull/bear: follows 12h trend only when confirmed by volume spike.
# Uses 12h EMA to reduce whipsaw vs shorter EMAs, volume filter to avoid false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 2.0 x 20-period average (4h) for significance
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h EMA (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filters
        price_above_ema = price > ema_50_12h_aligned[i]
        price_below_ema = price < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price above EMA with volume spike
            if price_above_ema and vol_filter:
                signals[i] = size
                position = 1
            # Short: price below EMA with volume spike
            elif price_below_ema and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA or volume drops
            if not price_above_ema or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA or volume drops
            if not price_below_ema or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_EMA50_VolumeSpike_12hTrend"
timeframe = "4h"
leverage = 1.0