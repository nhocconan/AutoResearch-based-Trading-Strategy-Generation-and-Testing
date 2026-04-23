#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Reversal with 1d EMA34 trend filter and volume spike.
- Williams %R(14) from 1d timeframe: oversold < -80, overbought > -20
- Long: Williams %R crosses above -80 (oversold bounce) + volume > 1.8x 20-period avg + price > 1d EMA34
- Short: Williams %R crosses below -20 (overbought rejection) + volume > 1.8x 20-period avg + price < 1d EMA34
- Exit: Williams %R crosses above -20 (long) or below -80 (short) OR 1d EMA34 trend flip
- Uses Williams %R for mean reversion signals, volume for conviction, 1d EMA34 for trend filter
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)
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
    
    # Volume confirmation: > 1.8x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (highest_high - close) / (highest_high - lowest_low) * -100
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral when range is zero
    )
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need 34 for EMA34, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(williams_r_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold bounce) + volume confirmation + price > 1d EMA34
            if (williams_r_aligned[i] > -80 and 
                williams_r_aligned[i-1] <= -80 and  # crossed above -80
                volume_confirm and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought rejection) + volume confirmation + price < 1d EMA34
            elif (williams_r_aligned[i] < -20 and 
                  williams_r_aligned[i-1] >= -20 and  # crossed below -20
                  volume_confirm and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR price < 1d EMA34 (trend flip)
            if (williams_r_aligned[i] > -20 and 
                williams_r_aligned[i-1] <= -20) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR price > 1d EMA34 (trend flip)
            if (williams_r_aligned[i] < -80 and 
                williams_r_aligned[i-1] >= -80) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Reversal_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0