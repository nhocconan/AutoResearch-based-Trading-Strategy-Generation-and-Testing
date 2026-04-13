#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot + volume spike + RSI mean reversion.
# Camarilla levels provide high-probability reversal points in ranging markets.
# Volume spike confirms institutional interest at key levels.
# RSI <30 for long, >70 for short to avoid buying strength/selling weakness.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        avg_gain[i] = np.mean(gain[i-13:i+1])
        avg_loss[i] = np.mean(loss[i-13:i+1])
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate average volume (20-period) for volume spike detection
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_h2 = np.zeros(len(close_1d))
    camarilla_l2 = np.zeros(len(close_1d))
    camarilla_h1 = np.zeros(len(close_1d))
    camarilla_l1 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        range_val = ph - pl
        
        camarilla_h4[i] = pc + range_val * 1.1/2
        camarilla_l4[i] = pc - range_val * 1.1/2
        camarilla_h3[i] = pc + range_val * 1.1/4
        camarilla_l3[i] = pc - range_val * 1.1/4
        camarilla_h2[i] = pc + range_val * 1.1/6
        camarilla_l2[i] = pc - range_val * 1.1/6
        camarilla_h1[i] = pc + range_val * 1.1/12
        camarilla_l1[i] = pc - range_val * 1.1/12
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi[i]) or np.isnan(avg_volume[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h2_aligned[i]) or np.isnan(l2_aligned[i]) or
            np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        rsi_val = rsi[i]
        
        # Volume spike: current volume > 2x average volume
        volume_spike = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: price at L3/L4 + volume spike + RSI oversold
            if ((abs(price - l3_aligned[i]) < 0.001 * price or 
                 abs(price - l4_aligned[i]) < 0.001 * price) and
                volume_spike and
                rsi_val < 30):
                position = 1
                signals[i] = position_size
            # Short: price at H3/H4 + volume spike + RSI overbought
            elif ((abs(price - h3_aligned[i]) < 0.001 * price or 
                   abs(price - h4_aligned[i]) < 0.001 * price) and
                  volume_spike and
                  rsi_val > 70):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H1/H2 or RSI overbought
            if (abs(price - h1_aligned[i]) < 0.001 * price or
                abs(price - h2_aligned[i]) < 0.001 * price or
                rsi_val > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L1/L2 or RSI oversold
            if (abs(price - l1_aligned[i]) < 0.001 * price or
                abs(price - l2_aligned[i]) < 0.001 * price or
                rsi_val < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Volume_RSI_v1"
timeframe = "12h"
leverage = 1.0