#!/usr/bin/env python3
name = "6h_ThreeLineBreak_1dTrend_Volume"
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
    
    # Load daily data ONCE before loop for trend filter and Three Line Break
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Three Line Break (TLB) from daily closes
    daily_close = df_1d['close'].values
    tlb_dir = np.zeros(len(daily_close))  # 1: up, -1: down, 0: no change
    line_heights = []  # store closing prices of each line
    current_line = 0
    
    # Initialize with first close
    if len(daily_close) > 0:
        line_heights.append(daily_close[0])
    
    for i in range(1, len(daily_close)):
        if daily_close[i] > line_heights[-1]:
            # Upward reversal: need to exceed previous line
            if len(line_heights) >= 3:
                # Check against the line 3 steps back
                if daily_close[i] > line_heights[-3]:
                    line_heights.append(daily_close[i])
                    current_line += 1
                    tlb_dir[i] = 1
                else:
                    tlb_dir[i] = 0
            else:
                line_heights.append(daily_close[i])
                current_line += 1
                tlb_dir[i] = 1
        elif daily_close[i] < line_heights[-1]:
            # Downward reversal: need to go below previous line
            if len(line_heights) >= 3:
                # Check against the line 3 steps back
                if daily_close[i] < line_heights[-3]:
                    line_heights.append(daily_close[i])
                    current_line += 1
                    tlb_dir[i] = -1
                else:
                    tlb_dir[i] = 0
            else:
                line_heights.append(daily_close[i])
                current_line += 1
                tlb_dir[i] = -1
        else:
            tlb_dir[i] = 0
    
    # Align TLB direction to 6h timeframe
    tlb_dir_aligned = align_htf_to_ltf(prices, df_1d, tlb_dir)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(tlb_dir_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TLB up with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if tlb_dir_aligned[i] == 1 and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: TLB down with volume and daily downtrend
            elif tlb_dir_aligned[i] == -1 and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TLB reverses down or volume drops
            if tlb_dir_aligned[i] == -1 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TLB reverses up or volume drops
            if tlb_dir_aligned[i] == 1 or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Three Line Break (TLB) from daily chart with 1d trend and volume confirmation
# - TLB filters out minor price movements and only reverses on significant price action
# - Entry when TLB shows new line in direction of daily trend with volume confirmation
# - Volume spike (1.8x average) confirms institutional participation in the move
# - Works in both bull (buy TLB up in uptrend) and bear (sell TLB down in downtrend)
# - Exit when TLB reverses direction or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual daily TLB (not 6h) for better signal quality
# - Daily trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: TLB (1d) + trend (1d) + volume (6h) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits