#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d Supertrend trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold), price > 1d Supertrend (uptrend), and volume > 1.8x 20-bar avg.
# Short when Williams %R > -20 (overbought), price < 1d Supertrend (downtrend), and volume > 1.8x 20-bar avg.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses proven mean reversion logic with strict trend filter to avoid false signals in strong trends.
# Volume confirmation (1.8x) limits overtrading while maintaining sufficient trade frequency.
# Williams %R is effective in ranging markets (common in 2025 BTC/ETH) while Supertrend filters trend direction.

name = "6h_WilliamsR_MeanRev_1dSupertrend_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Supertrend trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Supertrend for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(high_1d).shift(1) - pd.Series(close_1d)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean()
    
    # Supertrend calculation
    upper_band = ((high_1d + low_1d) / 2) + (3 * atr_1d)
    lower_band = ((high_1d + low_1d) / 2) - (3 * atr_1d)
    
    supertrend = np.full(len(close_1d), np.nan)
    direction = np.full(len(close_1d), 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1d)):
        if np.isnan(supertrend[i-1]):
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            if close_1d[i] > supertrend[i-1]:
                direction[i] = 1
            else:
                direction[i] = -1
            
            if direction[i] == 1:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align 1d Supertrend to 6h timeframe (completed 1d bar only)
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_1d_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Williams %R (14-period) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and Supertrend
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(williams_r.iloc[i]) or 
            np.isnan(supertrend_1d_aligned[i]) or 
            np.isnan(direction_1d_aligned[i]) or 
            np.isnan(volume_confirm.iloc[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r.iloc[i]
        curr_supertrend = supertrend_1d_aligned[i]
        curr_direction = direction_1d_aligned[i]
        curr_volume_confirm = volume_confirm.iloc[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R oversold (< -80), uptrend (direction=1), volume spike
            if (curr_williams_r < -80 and 
                curr_direction == 1 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), downtrend (direction=-1), volume spike
            elif (curr_williams_r > -20 and 
                  curr_direction == -1 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R crosses above -50 (mean reversion complete)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R crosses below -50 (mean reversion complete)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals