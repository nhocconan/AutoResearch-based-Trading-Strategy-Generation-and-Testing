#!/usr/bin/env python3
# 4H_ThreeLineBreak_12hTrend_Volume
# Hypothesis: Three Line Break (TLB) reversal signals combined with 12h EMA trend filter and volume confirmation.
# TLB filters out small reversals, focusing on significant trend changes. Works in both bull and bear markets by
# only taking trades in the direction of the 12h trend, reducing whipsaws. Volume > 2.0x average confirms momentum.
# Target: 20-40 trades/year to minimize fee drag.

name = "4H_ThreeLineBreak_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def three_line_break(close):
    """Calculate Three Line Break reversal signals.
    Returns: 1 for bullish reversal, -1 for bearish reversal, 0 for no signal.
    """
    n = len(close)
    tlb = np.zeros(n, dtype=int)
    
    if n < 4:
        return tlb
    
    # Initialize first line
    line_high = close[0]
    line_low = close[0]
    line_count = 1
    reversal = 0  # 1 for up, -1 for down
    
    for i in range(1, n):
        if close[i] > line_high:
            # Continue upward line
            line_high = close[i]
            line_low = close[i]
            reversal = 1
        elif close[i] < line_low:
            # Continue downward line
            line_high = close[i]
            line_low = close[i]
            reversal = -1
        else:
            # Check for reversal
            if reversal == 1 and close[i] < line_low:
                # Bearish reversal: close below prior low
                tlb[i] = -1
                line_high = close[i]
                line_low = close[i]
                line_count += 1
                reversal = -1
            elif reversal == -1 and close[i] > line_high:
                # Bullish reversal: close above prior high
                tlb[i] = 1
                line_high = close[i]
                line_low = close[i]
                line_count += 1
                reversal = 1
    
    return tlb

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Three Line Break signals
    tlb_signals = three_line_break(close)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_12h_aligned[i]
        price_below_ema = close[i] < ema_12h_aligned[i]
        
        if position == 0:
            # Long entry: TLB bullish reversal + above 12h EMA + volume spike
            if (tlb_signals[i] == 1 and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: TLB bearish reversal + below 12h EMA + volume spike
            elif (tlb_signals[i] == -1 and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: TLB bearish reversal or price below EMA
            if (tlb_signals[i] == -1 or not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: TLB bullish reversal or price above EMA
            if (tlb_signals[i] == 1 or not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals