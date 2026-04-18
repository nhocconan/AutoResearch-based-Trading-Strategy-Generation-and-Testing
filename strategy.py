#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA34_Volume
Hypothesis: 4-hour breakouts above Camarilla R1 or below S1 with 12-hour EMA34 trend filter and volume confirmation.
Camarilla levels provide precise intraday support/resistance, EMA34 filters trend direction, volume confirms breakout strength.
Designed for low trade frequency (target: 20-50/year) with strong performance in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12-hour EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 with proper smoothing
    ema34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema34_12h[33] = np.mean(close_12h[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_12h)):
            ema34_12h[i] = close_12h[i] * alpha + ema34_12h[i-1] * (1 - alpha)
    
    # Align 12-hour EMA34 to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's OHLC (assuming daily data available from 12h aggregation)
        # For 4h data, we need to look back to previous day's values
        # Since we don't have daily data directly, we'll approximate using 12h data
        # This is a simplification - in practice would use actual daily OHLC
        if i >= 3:  # Approximate daily lookback
            prev_high = np.max(high[i-3:i])  # Approximate daily high
            prev_low = np.min(low[i-3:i])    # Approximate daily low
            prev_close = close[i-1]          # Previous close
            range_val = prev_high - prev_low
            camarilla_r1[i] = prev_close + range_val * 1.1 / 12
            camarilla_s1[i] = prev_close - range_val * 1.1 / 12
    
    # Volume spike: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 3)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Camarilla R1 with volume spike and 12h uptrend
            if (close[i] > camarilla_r1[i] and vol_spike[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S1 with volume spike and 12h downtrend
            elif (close[i] < camarilla_s1[i] and vol_spike[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Camarilla S1 or 12h trend turns down
            if (close[i] < camarilla_s1[i] or close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Camarilla R1 or 12h trend turns up
            if (close[i] > camarilla_r1[i] or close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0