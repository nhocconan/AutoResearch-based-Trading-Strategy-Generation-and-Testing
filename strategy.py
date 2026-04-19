#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-day RSI filter.
# Long when: Alligator jaws (13-period SMMA) above teeth (8-period SMMA) and teeth above lips (5-period SMMA),
#            and 1-day RSI > 50.
# Short when: Jaws below teeth and teeth below lips, and 1-day RSI < 50.
# Exit when: Alligator lines re-cross (jaws crosses teeth).
# Williams Alligator identifies trend alignment, while 1-day RSI filters for momentum bias.
# Target: 20-35 trades/year per symbol. Works in trending markets (both bull and bear).
name = "6h_WilliamsAlligator_1dRSI_Filter"
timeframe = "6h"
leverage = 1.0

def smma(src, length):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if length < 1:
        return src
    result = np.full_like(src, np.nan, dtype=float)
    # First value is simple average
    result[length-1] = np.mean(src[:length])
    # Subsequent values: (prev * (length-1) + current) / length
    for i in range(length, len(src)):
        result[i] = (result[i-1] * (length-1) + src[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # 1-day data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 6h close
    # Jaws: 13-period SMMA of SMMA (8 periods offset)
    # Teeth: 8-period SMMA of SMMA (5 periods offset)
    # Lips: 5-period SMMA
    smma5 = smma(close, 5)
    smma8 = smma(close, 8)
    smma13 = smma(close, 13)
    
    jaws = smma(smma13, 8)  # 13-period SMMA smoothed with 8-period
    teeth = smma(smma8, 5)   # 8-period SMMA smoothed with 5-period
    lips = smma5              # 5-period SMMA
    
    # Calculate 1-day RSI (14-period)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    # Wilder's smoothing for RSI
    avg_gain[13] = np.mean(gain[1:14])  # First average gain
    avg_loss[13] = np.mean(loss[1:14])  # First average loss
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1-day data to 6h timeframe
    jaws_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaws)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for all indicators to be calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaws_val = jaws_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        
        if position == 0:
            # Long entry: Jaws > Teeth > Lips (bullish alignment) and RSI > 50
            if (jaws_val > teeth_val and teeth_val > lips_val and rsi_val > 50):
                signals[i] = 0.25
                position = 1
            # Short entry: Jaws < Teeth < Lips (bearish alignment) and RSI < 50
            elif (jaws_val < teeth_val and teeth_val < lips_val and rsi_val < 50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Jaws crosses below Teeth (trend weakening)
            if jaws_val < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Jaws crosses above Teeth (trend weakening)
            if jaws_val > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals