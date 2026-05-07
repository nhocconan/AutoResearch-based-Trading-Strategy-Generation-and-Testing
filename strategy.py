#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeConfirmation
Hypothesis: Use Donchian channel (20-period high/low) breakout on 12h timeframe, filtered by 1d EMA trend and volume spike. Go long when price breaks above upper band in 1d uptrend with volume >1.5x average. Go short when price breaks below lower band in 1d downtrend with volume >1.5x average. Exit when price crosses opposite Donchian band or trend reverses. Designed for 12h to capture major trend moves with low frequency (target 15-35 trades/year).
"""

name = "12h_Donchian20_Breakout_1dTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Donchian channel (20-period) on 12h
    period = 20
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    for i in range(period, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume confirmation: 20-period average on 12h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period, 34)  # Warmup for Donchian and 1d EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(close_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 1d trend determination
        trend_1d_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_1d_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band in 1d uptrend with volume confirmation
            if (close[i] > upper[i] and 
                trend_1d_up and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band in 1d downtrend with volume confirmation
            elif (close[i] < lower[i] and 
                  trend_1d_down and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower Donchian band or 1d trend turns down
            if (close[i] < lower[i] or not trend_1d_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper Donchian band or 1d trend turns up
            if (close[i] > upper[i] or not trend_1d_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals