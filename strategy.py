#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover (20/50) with 4h trend filter and volume confirmation.
# Works in bull (EMA20 > EMA50 + 4h trend up) and bear (EMA20 < EMA50 + 4h trend down).
# Target: 15-37 trades/year (60-150 total over 4 years) to avoid fee drag.
# Uses 4h EMA50 for trend direction, 1h EMA20/50 for entry timing, volume filter to avoid noise.
name = "1h_EMA20_50_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h EMAs for entry
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for EMA calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        ema_20_val = ema_20[i]
        ema_50_val = ema_50[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: EMA20 > EMA50, 4h trend up (price > 4h EMA50), volume confirmation
            if ema_20_val > ema_50_val and close[i] > ema_50_4h_val and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: EMA20 < EMA50, 4h trend down (price < 4h EMA50), volume confirmation
            elif ema_20_val < ema_50_val and close[i] < ema_50_4h_val and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: EMA20 crosses below EMA50 or volume filter fails
            if ema_20_val < ema_50_val or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: EMA20 crosses above EMA50 or volume filter fails
            if ema_20_val > ema_50_val or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals