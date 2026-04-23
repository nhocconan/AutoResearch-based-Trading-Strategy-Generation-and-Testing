#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d EMA50 Trend and Volume Spike Filter
- Williams %R identifies overbought/oversold conditions: long when %R crosses above -80 from below, short when crosses below -20 from above
- 1d EMA(50) ensures alignment with higher timeframe trend for multi-timeframe confirmation
- Volume > 2.0x 20-period average confirms strong momentum and reduces false signals
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in bull markets via trend-following breakouts, in bear markets via mean reversion from extremes
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
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14 period) on 6h data
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 20)  # Williams %R, EMA1d, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R signals with trend filter and volume spike
        # Long: %R crosses above -80 from below + uptrend + volume spike
        # Short: %R crosses below -20 from above + downtrend + volume spike
        williams_r_prev = williams_r[i-1] if i > 0 else -100
        
        long_cross = williams_r_prev <= -80 and williams_r[i] > -80
        short_cross = williams_r_prev >= -20 and williams_r[i] < -20
        
        long_signal = long_cross and (close[i] > ema_50_1d_aligned[i]) and (volume[i] > 2.0 * vol_ma[i])
        short_signal = short_cross and (close[i] < ema_50_1d_aligned[i]) and (volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Williams %R extreme or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: %R crosses above -20 (overbought) or trend reversal
                if (williams_r_prev >= -20 and williams_r[i] < -20) or (close[i] < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: %R crosses below -80 (oversold) or trend reversal
                if (williams_r_prev <= -80 and williams_r[i] > -80) or (close[i] > ema_50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0