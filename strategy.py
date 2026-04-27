#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d trend filter.
- Williams %R identifies overbought/oversold conditions on 12h
- 1d EMA34 filters for trend direction (trade with trend)
- Volume spike confirms momentum (volume > 1.8x 20-period avg)
- Target: 15-25 trades/year to minimize fee drag
- Uses discrete position sizing (0.25) to minimize churn
- Works in bull/bear: mean reversion in ranges, trend filter avoids counter-trend in strong moves
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R (14-period)
    highest_high = np.full(len(high_12h), np.nan)
    lowest_low = np.full(len(low_12h), np.nan)
    williams_r = np.full(len(close_12h), np.nan)
    
    for i in range(len(high_12h)):
        if i >= 13:  # 14-period lookback
            highest_high[i] = np.max(high_12h[i-13:i+1])
            lowest_low[i] = np.min(low_12h[i-13:i+1])
            if highest_high[i] != lowest_low[i]:
                williams_r[i] = -100 * (highest_high[i] - close_12h[i]) / (highest_high[i] - lowest_low[i])
    
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on daily close
    ema_34 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        # Seed with SMA
        ema_34[33] = np.mean(close_1d[:34])
        # EMA calculation
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34[i] = alpha * close_1d[i] + (1 - alpha) * ema_34[i-1]
    
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(40, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + price above EMA34 (uptrend) + volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + price below EMA34 (downtrend) + volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Williams %R overbought (> -20) OR price crosses below EMA34
            if (williams_r_aligned[i] > -20 or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R oversold (< -80) OR price crosses above EMA34
            if (williams_r_aligned[i] < -80 or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_EMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0