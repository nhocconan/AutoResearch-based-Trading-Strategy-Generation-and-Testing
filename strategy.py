#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with weekly trend filter and volume spike filter.
- Long when Williams %R < -80 (oversold) + price > 1w EMA200 (uptrend) + volume spike (>1.5x avg)
- Short when Williams %R > -20 (overbought) + price < 1w EMA200 (downtrend) + volume spike (>1.5x avg)
- Exit when Williams %R crosses back to neutral (-50) or trend reverses
- Williams %R is effective in ranging markets; weekly EMA200 filters for major trend
- Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 6h Williams %R(14)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(13, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * ((highest_high - close) / (highest_high - lowest_low)), 
                          -50)
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for Williams %R and EMA
    start_idx = max(20, 14, 200)
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: Williams %R oversold + above 1w EMA200 + volume spike
            if (williams_r[i] < -80 and close[i] > ema200_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought + below 1w EMA200 + volume spike
            elif (williams_r[i] > -20 and close[i] < ema200_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Williams %R crosses above -50 (mean reversion) OR below 1w EMA200 (trend change)
            if (williams_r[i] > -50 or close[i] < ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -50 (mean reversion) OR above 1w EMA200 (trend change)
            if (williams_r[i] < -50 or close[i] > ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_1wEMA200_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0