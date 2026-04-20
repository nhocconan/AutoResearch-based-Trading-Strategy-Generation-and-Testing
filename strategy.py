#!/usr/bin/env python3
# 6h_1d_LiquidityVoid_Reversal_Scalp
# Hypothesis: Trade mean-reversion at 1d liquidity voids (unfilled gaps) on 6h timeframe.
# Liquidity voids occur when price gaps overnight and leaves unfilled volume.
# Price tends to return to fill these voids, creating mean-reversion opportunities.
# Uses volume confirmation and volatility filter to avoid whipsaws.
# Targets 15-35 trades/year by requiring void identification and volume confirmation.

name = "6h_1d_LiquidityVoid_Reversal_Scalp"
timeframe = "6h"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d liquidity voids (unfilled gaps)
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Identify gaps: overnight gap from previous close to current open
    gap_up = open_1d[1:] - close_1d[:-1]  # positive = gap up
    gap_down = close_1d[:-1] - open_1d[1:]  # positive = gap down
    
    # Create arrays aligned with 1d index (same length as close_1d)
    gap_up_full = np.concatenate([[0], gap_up])
    gap_down_full = np.concatenate([[0], gap_down])
    
    # Define liquidity void areas (unfilled gaps)
    # For gap up: void is between previous close and current open
    # For gap down: void is between current open and previous close
    void_high = np.maximum(open_1d, close_1d)  # higher of open/close
    void_low = np.minimum(open_1d, close_1d)   # lower of open/close
    
    # Only consider significant gaps (>0.1% of price)
    min_gap = 0.001 * void_high
    significant_gap = (void_high - void_low) > min_gap
    
    # Align void levels to 6h timeframe
    void_high_aligned = align_htf_to_ltf(prices, df_1d, void_high)
    void_low_aligned = align_htf_to_ltf(prices, df_1d, void_low)
    significant_gap_aligned = align_htf_to_ltf(prices, df_1d, significant_gap.astype(float))
    
    # Calculate 1d ATR for volatility filter (14-period)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(void_high_aligned[i]) or np.isnan(void_low_aligned[i]) or 
            np.isnan(significant_gap_aligned[i]) or np.isnan(atr_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Skip if not a significant gap
        if significant_gap_aligned[i] < 0.5:
            if position != 0:
                signals[i] = 0.25 if position == 1 else -0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for mean reversion to fill the void
            # Long when price approaches void low from below (filling gap down)
            if (close[i] <= void_low_aligned[i] * 1.002 and  # within 0.2% of void low
                close[i] >= void_low_aligned[i] * 0.998 and
                volume[i] > 1.5 * volume_ma[i]):  # volume confirmation
                signals[i] = 0.25
                position = 1
            # Short when price approaches void high from above (filling gap up)
            elif (close[i] >= void_high_aligned[i] * 0.998 and  # within 0.2% of void high
                  close[i] <= void_high_aligned[i] * 1.002 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price reaches void high (gap filled) or reverses
            if (close[i] >= void_high_aligned[i] * 0.998 or  # reached void high
                (close[i] <= void_low_aligned[i] * 1.005 and  # reversed back down
                 volume[i] > volume_ma[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches void low (gap filled) or reverses
            if (close[i] <= void_low_aligned[i] * 1.002 or  # reached void low
                (close[i] >= void_high_aligned[i] * 0.995 and  # reversed back up
                 volume[i] > volume_ma[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals