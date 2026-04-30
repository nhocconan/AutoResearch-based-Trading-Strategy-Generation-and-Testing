#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversal with 1d Supertrend trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold), price > 1d Supertrend (uptrend), and volume > 1.5x 20-bar avg.
# Short when Williams %R > -20 (overbought), price < 1d Supertrend (downtrend), and volume > 1.5x 20-bar avg.
# Exit when Williams %R crosses -50 (mean reversion to midpoint).
# Williams %R identifies overextended moves, Supertrend filters for trend alignment, volume confirms momentum.
# Timeframe: 6h as per experiment guidelines.

name = "6h_WilliamsR_MeanRev_1dSupertrend_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Supertrend trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Supertrend for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and ATR for Supertrend
    tr1 = pd.Series(high_1d).rolling(2).max() - pd.Series(low_1d).rolling(2).min()
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/10, adjust=False).mean()  # ATR(10)
    
    # Supertrend calculation
    upper_band = (pd.Series(high_1d) + pd.Series(low_1d)) / 2 + 3 * atr
    lower_band = (pd.Series(high_1d) + pd.Series(low_1d)) / 2 - 3 * atr
    
    supertrend = np.zeros(len(close_1d))
    direction = np.ones(len(close_1d))  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band.iloc[0] if len(upper_band) > 0 else 0
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > upper_band.iloc[i-1]:
            direction[i] = 1
        elif close_1d[i] < lower_band.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            if direction[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i-1]
        
        supertrend[i] = lower_band.iloc[i] if direction[i] == 1 else upper_band.iloc[i]
    
    # Align Supertrend to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Calculate Williams %R on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = williams_r.values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for Williams %R and Supertrend
    
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
            # Long: Williams %R oversold (< -80), uptrend (direction=1), volume confirmation
            if (curr_williams_r < -80 and 
                curr_direction == 1 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20), downtrend (direction=-1), volume confirmation
            elif (curr_williams_r > -20 and 
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