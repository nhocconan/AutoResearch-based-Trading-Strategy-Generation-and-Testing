#!/usr/bin/env python3
"""
4h_ThreeLineBreak_Reversal_Confirmation
Hypothesis: Three Line Break (TLB) reversal signals filtered by 1w EMA trend and volume spike, with ATR-based exit. 
TLB captures momentum shifts; 1w EMA ensures alignment with higher timeframe trend; volume confirms institutional participation. 
Works in both bull/bear by following trend direction. Target: 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Three Line Break (TLB)
    # TLB: new line when price moves beyond prior 3 lines in opposite direction
    tblb_direction = np.zeros(n)  # 1=up, -1=down, 0=no change
    line_high = np.full(n, np.nan)
    line_low = np.full(n, np.nan)
    
    if n > 0:
        line_high[0] = high[0]
        line_low[0] = low[0]
        tblb_direction[0] = 0
        
        reversal_count = 0
        last_dir = 0
        
        for i in range(1, n):
            if last_dir >= 0:  # in up trend or neutral
                if low[i] < line_low[i-1]:
                    reversal_count += 1
                    if reversal_count >= 3:
                        tblb_direction[i] = -1
                        line_high[i] = high[i]
                        line_low[i] = low[i]
                        last_dir = -1
                        reversal_count = 0
                    else:
                        tblb_direction[i] = last_dir
                        line_high[i] = max(line_high[i-1], high[i])
                        line_low[i] = line_low[i-1]
                else:
                    tblb_direction[i] = last_dir
                    line_high[i] = max(line_high[i-1], high[i])
                    line_low[i] = min(line_low[i-1], low[i])
                    reversal_count = 0
            else:  # in down trend
                if high[i] > line_high[i-1]:
                    reversal_count += 1
                    if reversal_count >= 3:
                        tblb_direction[i] = 1
                        line_high[i] = high[i]
                        line_low[i] = low[i]
                        last_dir = 1
                        reversal_count = 0
                    else:
                        tblb_direction[i] = last_dir
                        line_high[i] = line_high[i-1]
                        line_low[i] = min(line_low[i-1], low[i])
                else:
                    tblb_direction[i] = last_dir
                    line_high[i] = line_high[i-1]
                    line_low[i] = min(line_low[i-1], low[i])
                    reversal_count = 0
    
    # Multi-timeframe: 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: >2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 30)  # Warmup for EMA and volume
    
    for i in range(start_idx, n):
        if (np.isnan(tblb_direction[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        tlb_dir = tblb_direction[i]
        ema50 = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: TLB up + volume spike + above weekly EMA
            if tlb_dir == 1 and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: TLB down + volume spike + below weekly EMA
            elif tlb_dir == -1 and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TLB reversal down OR price crosses below weekly EMA
            if tlb_dir == -1:
                signals[i] = 0.0
                position = 0
            elif price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TLB reversal up OR price crosses above weekly EMA
            if tlb_dir == 1:
                signals[i] = 0.0
                position = 0
            elif price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_ThreeLineBreak_Reversal_Confirmation"
timeframe = "4h"
leverage = 1.0