#!/usr/bin/env python3
"""
4h_ThreeLineBreak_Reversal_Detection_v1
Hypothesis: Detect short-term reversions at key levels using 3-line break reversals combined with volume confirmation and 1d trend filter. Works in both bull/bear markets by capturing mean reversion moves after overextended moves. Target: 20-30 trades/year via strict 3-line break + volume confirmation requirement.
"""

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
    
    # Calculate 3-line break (3LB) reversals
    # Track the 3-line break direction and detect reversals
    line_breaks = np.zeros(n)
    line_breaks[0] = 1  # Start with up
    
    # For 3LB, we need to track the last closing price that made a new line
    line_break_levels = np.full(n, np.nan)
    line_break_levels[0] = close[0]
    
    current_direction = 1  # 1 for up, -1 for down
    last_line_close = close[0]
    
    for i in range(1, n):
        if current_direction == 1:  # Currently in up trend
            if close[i] < last_line_close - 2 * (high[i] - low[i]):  # Reverse down
                current_direction = -1
                line_breaks[i] = -1
                last_line_close = close[i]
            elif close[i] > last_line_close:  # Continue up, make new line if significant
                if close[i] > last_line_close + (high[i] - low[i]) * 0.5:  # New line threshold
                    line_breaks[i] = 1
                    last_line_close = close[i]
                else:
                    line_breaks[i] = 0
            else:
                line_breaks[i] = 0
        else:  # Currently in down trend
            if close[i] > last_line_close + 2 * (high[i] - low[i]):  # Reverse up
                current_direction = 1
                line_breaks[i] = 1
                last_line_close = close[i]
            elif close[i] < last_line_close:  # Continue down, make new line if significant
                if close[i] < last_line_close - (high[i] - low[i]) * 0.5:  # New line threshold
                    line_breaks[i] = -1
                    last_line_close = close[i]
                else:
                    line_breaks[i] = 0
            else:
                line_breaks[i] = 0
    
    # Detect 3-line break reversals (when line_breaks changes sign)
    reversal_signal = np.zeros(n)
    for i in range(1, n):
        if line_breaks[i] != 0 and line_breaks[i-1] != 0 and line_breaks[i] != line_breaks[i-1]:
            reversal_signal[i] = line_breaks[i]  # 1 for bullish reversal, -1 for bearish
    
    # 1d trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50 = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        rev_signal = reversal_signal[i]
        
        if position == 0:
            # Long: bullish 3LB reversal with volume and above 1d EMA
            if rev_signal == 1 and vol_spike and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: bearish 3LB reversal with volume and below 1d EMA
            elif rev_signal == -1 and vol_spike and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: bearish 3LB reversal or price below 1d EMA
            if rev_signal == -1 or price < ema_50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: bullish 3LB reversal or price above 1d EMA
            if rev_signal == 1 or price > ema_50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_ThreeLineBreak_Reversal_Detection_v1"
timeframe = "4h"
leverage = 1.0