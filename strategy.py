#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d volume spike and 4h RSI filter.
# Camarilla levels act as reversal zones in ranging markets.
# Volume spikes confirm institutional interest at key levels.
# RSI filter ensures we trade in the direction of momentum.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla and volume analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h5 = np.zeros(len(close_1d))
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    camarilla_l5 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        # Previous day's values
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rang = ph - pl
        
        camarilla_h5[i] = pc + 1.1 * rang / 2
        camarilla_h4[i] = pc + 1.1 * rang / 4
        camarilla_h3[i] = pc + 1.1 * rang / 6
        camarilla_l3[i] = pc - 1.1 * rang / 6
        camarilla_l4[i] = pc - 1.1 * rang / 4
        camarilla_l5[i] = pc - 1.1 * rang / 2
    
    # Align Camarilla levels to 4h timeframe
    h5_4h = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_4h = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Calculate 20-period average volume for spike detection
    vol_avg = np.zeros(n)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    # Calculate 4h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros(n)
    rsi = np.zeros(n)
    for i in range(14, n):
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(h5_4h[i]) or np.isnan(l5_4h[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = vol_avg[i]
        rsi_val = rsi[i]
        
        # Volume spike: current volume > 2.0x average volume
        volume_spike = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long reversal: price touches L3/L4 + volume spike + RSI < 40 (oversold bounce)
            if (abs(price - l3_4h[i]) < 0.001 * price or abs(price - l4_4h[i]) < 0.001 * price) and \
               volume_spike and \
               rsi_val < 40:
                position = 1
                signals[i] = position_size
            # Short reversal: price touches H3/H4 + volume spike + RSI > 60 (overbought reversal)
            elif (abs(price - h3_4h[i]) < 0.001 * price or abs(price - h4_4h[i]) < 0.001 * price) and \
                 volume_spike and \
                 rsi_val > 60:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 or RSI > 70
            if (price >= h3_4h[i] or rsi_val > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 or RSI < 30
            if (price <= l3_4h[i] or rsi_val < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Volume_Spike_RSI_v1"
timeframe = "4h"
leverage = 1.0