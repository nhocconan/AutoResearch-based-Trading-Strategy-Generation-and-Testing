#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d Supertrend trend filter and volume confirmation.
# Long when %R < -80 (oversold), price > 1d Supertrend (uptrend), and volume > 1.5x 20-bar avg.
# Short when %R > -20 (overbought), price < 1d Supertrend (downtrend), and volume > 1.5x 20-bar avg.
# Exit when %R reverts to -50 (mean reversion) or trend filter fails.
# Uses 1d Supertrend for higher timeframe trend alignment, targeting 12-30 trades/year on 12h.
# Williams %R captures short-term extremes, Supertrend avoids counter-trend trades, volume confirmation reduces false signals.
# Works in bull markets via mean reversion longs in uptrends and in bear markets via mean reversion shorts in downtrends.

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
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.ewm(alpha=1/10, adjust=False).mean()
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upperband = hl2 + (3.0 * atr_1d)
    lowerband = hl2 - (3.0 * atr_1d)
    
    supertrend = np.full_like(close_1d, np.nan, dtype=float)
    direction = np.full_like(close_1d, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_1d[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
    
    supertrend_1d = supertrend
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and Supertrend
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(supertrend_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_supertrend_1d = supertrend_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80), price > 1d Supertrend (uptrend), volume spike
            if (curr_williams_r < -80 and 
                curr_close > curr_supertrend_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), price < 1d Supertrend (downtrend), volume spike
            elif (curr_williams_r > -20 and 
                  curr_close < curr_supertrend_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: Williams %R reverts to -50 (mean reversion) OR trend fails
            if (curr_williams_r >= -50) or (curr_close <= curr_supertrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Williams %R reverts to -50 (mean reversion) OR trend fails
            if (curr_williams_r <= -50) or (curr_close >= curr_supertrend_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals