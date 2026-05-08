#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Chaikin_Pullback_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Chaikin Money Flow (21-period)
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    mf_multiplier = np.where(high == low, 0, mf_multiplier)
    mf_volume = mf_multiplier * volume
    cmf = pd.Series(mf_volume).rolling(window=21, min_periods=21).sum().values / \
          pd.Series(volume).rolling(window=21, min_periods=21).sum().values
    
    # 6-period EMA for pullback identification
    ema_6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(cmf[i]) or 
            np.isnan(ema_6[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback to EMA6 in uptrend with positive CMF and volume spike
            long_cond = (close[i] > ema_34_1d_aligned[i] and      # above daily trend
                        close[i] <= ema_6[i] * 1.01 and           # near or slightly below 6 EMA (pullback)
                        close[i] >= ema_6[i] * 0.99 and           # near or slightly above 6 EMA (pullback)
                        cmf[i] > 0.05 and                         # strong buying pressure
                        volume_spike[i])
            
            # Short: pullback to EMA6 in downtrend with negative CMF and volume spike
            short_cond = (close[i] < ema_34_1d_aligned[i] and   # below daily trend
                         close[i] >= ema_6[i] * 0.99 and        # near or slightly above 6 EMA (pullback)
                         close[i] <= ema_6[i] * 1.01 and        # near or slightly below 6 EMA (pullback)
                         cmf[i] < -0.05 and                     # strong selling pressure
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA6 or CMF turns negative
            if close[i] < ema_6[i] or cmf[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above EMA6 or CMF turns positive
            if close[i] > ema_6[i] or cmf[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals