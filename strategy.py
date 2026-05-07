#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R (14) with 1d trend filter and volume spike.
# Williams %R identifies overbought/oversold conditions. In trending markets,
# pullbacks to extreme levels offer high-probability entries. Uses 1d EMA for
# trend direction and volume spike for confirmation. Designed to work in both
# bull and bear markets by following the higher timeframe trend.
# Target: 20-40 trades/year per symbol to minimize fee drag.
name = "4h_WilliamsR_14_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend and Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume average for spike detection
    vol_avg_1d = pd.Series(df_1d['volume']).ewm(span=34, adjust=False, min_periods=34).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Williams %R calculation on 1d data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100,
                          -50)  # neutral when range is zero
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # 4h volume spike: current volume > 1.6x 34-period EMA
    vol_ema_4h = pd.Series(volume).ewm(span=34, adjust=False, min_periods=34).mean().values
    vol_spike = np.where(vol_ema_4h > 0, volume / vol_ema_4h, 1.0) > 1.6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for Williams %R
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume spike in uptrend
            long_condition = (williams_r_aligned[i] < -80) and vol_spike[i] and uptrend
            # Short: Williams %R overbought (> -20) with volume spike in downtrend
            short_condition = (williams_r_aligned[i] > -20) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R returns above -50 or trend turns down
            if (williams_r_aligned[i] > -50) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R returns below -50 or trend turns up
            if (williams_r_aligned[i] < -50) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals