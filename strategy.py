#!/usr/bin/env python3
"""
6h Williams %R Extreme + 12h EMA Trend + Volume Spike
Hypothesis: Williams %R identifies overextended moves; fade extremes in direction of 12h EMA trend with volume confirmation.
Works in bull markets by buying pullbacks in uptrends and in bear markets by selling rallies in downtrends.
Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 15-25 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R(14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Williams %R, volume MA, and EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Williams %R < -80 (oversold) AND price above 12h EMA (uptrend) AND volume spike
            long_entry = (curr_williams_r < -80) and (curr_close > ema_trend) and vol_spike
            # Short: Williams %R > -20 (overbought) AND price below 12h EMA (downtrend) AND volume spike
            short_entry = (curr_williams_r > -20) and (curr_close < ema_trend) and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Williams %R > -20 (overbought) OR price crosses below 12h EMA
            if (curr_williams_r > -20) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Williams %R < -80 (oversold) OR price crosses above 12h EMA
            if (curr_williams_r < -80) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0