#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels on 12-hour chart combined with 1-day trend filter and volume confirmation provides high-probability breakout entries. Works in bull markets by catching uptrend continuations and in bear markets by catching short-term reversals with institutional volume. Uses discrete position sizing to minimize fee churn.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Calculate 12h Camarilla pivot levels (using previous day's OHLC)
    # For 12h chart, we use the previous 12h bar's OHLC to calculate pivots
    # Camarilla formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous period
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # First bar has no previous data
    prev_close[0] = close[0]
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 50-period EMA on daily for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.zeros_like(close_1d)
    ema_50_1d[:] = np.nan
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 49) / 51
    
    # Align 1d EMA to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = np.zeros_like(volume)
    vol_ma_20[:] = np.nan
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 + 1d uptrend (price > EMA50) + volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 1d downtrend (price < EMA50) + volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below S1 or loss of volume spike
            if (close[i] < camarilla_s1[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above R1 or loss of volume spike
            if (close[i] > camarilla_r1[i] or not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals