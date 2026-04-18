# 6h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Uses daily Camarilla pivot levels (R1/S1) with 1d EMA trend filter and volume confirmation.
# Designed for 60-120 trades/year to minimize fee drag. Works in both bull and bear via 1d trend filter.

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
    
    # Get daily data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    R1 = close_1d + range_hl * 1.1 / 12
    S1 = close_1d - range_hl * 1.1 / 12
    
    # Calculate EMA(34) on daily close
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2/35) + (ema_34_1d[i-1] * 33/35)
    
    # Align daily indicators to 6h timeframe
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h ATR(14)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[:14])
        else:
            atr[i] = (tr[i] * 1/14) + (atr[i-1] * 13/14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need EMA, ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or np.isnan(ema_34_1d_6h[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        vol_confirmed = volume[i] > 1.8 * vol_ma[i]
        
        # Trend filter: price above/below 1d EMA34
        trend_up = close[i] > ema_34_1d_6h[i]
        trend_down = close[i] < ema_34_1d_6h[i]
        
        if position == 0:
            # Long entry: close above R1 + 0.2*ATR, with volume and trend filter
            if (close[i] > R1_6h[i] + 0.2 * atr[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below S1 - 0.2*ATR, with volume and trend filter
            elif (close[i] < S1_6h[i] - 0.2 * atr[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below S1 or ATR-based stop
            if close[i] < S1_6h[i] - 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above R1 or ATR-based stop
            if close[i] > R1_6h[i] + 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0