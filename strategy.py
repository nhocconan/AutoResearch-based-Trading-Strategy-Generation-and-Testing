#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Reversal with 1d EMA50 trend filter and volume spike.
- Williams %R(14): momentum oscillator, long when crosses above -80 (oversold), short when crosses below -20 (overbought)
- Trend filter: price > 1d EMA50 for longs, price < 1d EMA50 for shorts
- Volume confirmation: > 1.8x 20-period average (tight to avoid overtrading)
- Exit: Williams %R crosses below -50 for longs, above -50 for shorts OR EMA50 trend flip
- Uses Williams %R for mean reversion entries, volume for conviction, 1d EMA50 for HTF filter
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend)
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
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Long signal: %R crosses above -80 (from below)
    # Short signal: %R crosses below -20 (from above)
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = ((highest_high - close) / (highest_high - lowest_low)) * -100
    
    # Volume confirmation: > 1.8x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 50)  # Need 14 for Williams %R, 20 for volume MA, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below) + volume confirmation + price > 1d EMA50
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                volume_confirm and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) + volume confirmation + price < 1d EMA50
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  volume_confirm and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (from above) OR price < 1d EMA50 (trend flip)
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (from below) OR price > 1d EMA50 (trend flip)
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Reversal_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0