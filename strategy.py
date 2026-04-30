#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d Supertrend trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold), price > 1d Supertrend, and volume > 1.5x 20-bar avg.
# Short when Williams %R > -20 (overbought), price < 1d Supertrend, and volume > 1.5x 20-bar avg.
# Exit when Williams %R crosses -50 (mean reversion).
# Uses 12h timeframe to reduce trade frequency (target: 12-37 trades/year) and avoid fee drag.
# 1d Supertrend provides higher timeframe trend alignment; volume confirmation reduces false signals.
# Works in bull markets via mean reversion longs and in bear markets via mean reversion shorts with trend alignment.

name = "12h_WilliamsR_MeanRev_1dSupertrend_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Supertrend for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR calculation for Supertrend
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=10, min_periods=10).mean()
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr_1d)
    lower_band = hl2 - (3.0 * atr_1d)
    
    supertrend = np.full(len(close_1d), np.nan)
    direction = np.full(len(close_1d), np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(close_1d)):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            continue
            
        if i == 10:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            if close_1d[i-1] > supertrend[i-1]:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
    
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    
    # Calculate Williams %R on 12h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_supertrend = supertrend_aligned[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80), price > Supertrend (uptrend), volume spike
            if (curr_williams_r < -80 and 
                curr_close > curr_supertrend and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price < Supertrend (downtrend), volume spike
            elif (curr_williams_r > -20 and 
                  curr_close < curr_supertrend and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R crosses above -50 (mean reversion)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R crosses below -50 (mean reversion)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals