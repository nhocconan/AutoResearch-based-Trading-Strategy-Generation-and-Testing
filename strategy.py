#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d Supertrend filter and volume confirmation.
# Long when: price breaks above upper BB(20,2) AND 1d Supertrend is bullish AND volume > 2x 20-bar avg.
# Short when: price breaks below lower BB(20,2) AND 1d Supertrend is bearish AND volume > 2x 20-bar avg.
# Exit when price crosses middle BB (20-period SMA) or Supertrend flips.
# Bollinger Squeeze identifies low volatility periods that often precede explosive moves.
# 1d Supertrend provides higher-timeframe trend filter to avoid counter-trend entries.
# Volume confirmation ensures breakout validity.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_BollingerSqueeze_Breakout_1dSupertrend_Volume_v1"
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
    
    # Calculate 1d Supertrend (ATR=10, mult=3.0)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr_1d)
    lower_band = hl2 - (3.0 * atr_1d)
    
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            supertrend[i] = lower_band[i]
            direction[i] = -1
            
        # Adjust bands
        if direction[i] == direction[i-1]:
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        else:
            if direction[i] == 1:
                upper_band[i] = hl2[i] + (3.0 * atr_1d[i])
            else:
                lower_band[i] = hl2[i] - (3.0 * atr_1d[i])
    
    # Align 1d Supertrend direction to 6h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Calculate Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    middle_bb = sma_20  # 20-period SMA
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 20)  # warmup for BB and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper BB AND 1d Supertrend bullish AND volume confirmation
            if (curr_high > upper_bb[i] and 
                supertrend_direction_aligned[i] == 1 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB AND 1d Supertrend bearish AND volume confirmation
            elif (curr_low < lower_bb[i] and 
                  supertrend_direction_aligned[i] == -1 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: price crosses below middle BB OR Supertrend flips bearish
            if (curr_close < middle_bb[i] or 
                supertrend_direction_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price crosses above middle BB OR Supertrend flips bullish
            if (curr_close > middle_bb[i] or 
                supertrend_direction_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals