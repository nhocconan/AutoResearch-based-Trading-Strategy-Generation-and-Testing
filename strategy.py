#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h Supertrend filter and volume confirmation.
# Long when Williams %R < -80 (oversold), price > 12h Supertrend (uptrend), and volume > 1.5x 20-bar avg.
# Short when Williams %R > -20 (overbought), price < 12h Supertrend (downtrend), and volume spike.
# Exit when Williams %R crosses above -50 (long) or below -50 (short) - mean reversion to midpoint.
# Uses proven mean reversion edge in ranging markets with trend filter to avoid counter-trend trades.
# 6h timeframe balances trade frequency and signal quality; volume confirmation reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.

name = "6h_WilliamsR_MeanRev_12hSupertrend_Trend_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for Supertrend trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3.0 * atr)
    lower_band = hl2 - (3.0 * atr)
    
    supertrend = np.full_like(close_12h, np.nan, dtype=float)
    direction = np.full_like(close_12h, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_12h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend and direction to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Williams %R and Supertrend
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_supertrend = supertrend_aligned[i]
        curr_direction = direction_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80), uptrend (direction=1), volume spike
            if (curr_williams_r < -80 and 
                curr_close > curr_supertrend and 
                curr_direction == 1 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), downtrend (direction=-1), volume spike
            elif (curr_williams_r > -20 and 
                  curr_close < curr_supertrend and 
                  curr_direction == -1 and 
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