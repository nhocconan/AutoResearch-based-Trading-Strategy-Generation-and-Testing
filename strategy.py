#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA50 Trend Filter + Volume Spike
# Williams %R(14) identifies overbought (> -20) and oversold (< -80) conditions.
# In strong trends, price can remain overbought/oversold, so we filter with 1d EMA50:
#   - Only take longs when Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend)
#   - Only take shorts when Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend)
# Volume spike confirms institutional participation at these extremes.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
name = "6h_WilliamsR_1dEMA50_Volume"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate EMA50 on 1d data for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 2.0 * 24-period average volume (4 days on 6h chart)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        wr = williams_r[i]
        ema_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price above EMA50 AND volume spike
            if wr < -80 and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below EMA50 AND volume spike
            elif wr > -20 and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns above -50 (momentum fading) or price below EMA50 (trend change)
            if wr > -50 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns below -50 (momentum fading) or price above EMA50 (trend change)
            if wr < -50 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals