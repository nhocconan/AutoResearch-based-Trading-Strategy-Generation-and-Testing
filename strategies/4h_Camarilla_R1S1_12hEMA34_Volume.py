#!/usr/bin/env python3
"""
Hypothesis: 4h-based strategy using Camarilla pivot levels (R1/S1) from 1d combined with 12h EMA(34) trend filter and volume confirmation. 
Camarilla levels provide high-probability reversal points in ranging markets while the 12h EMA filters for trend direction.
Volume confirmation ensures breakouts have conviction. Designed for 20-40 trades/year to minimize fee drag.
Works in bull markets (buy R1 breaks in uptrend) and bear markets (sell S1 breaks in downtrend).
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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            camarilla_r1[i] = close_1d[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
            camarilla_s1[i] = close_1d[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 12h data for EMA(34) trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h close
    ema_34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema_34_12h[33] = np.mean(close_12h[:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = (close_12h[i] * 2/35) + (ema_34_12h[i-1] * 33/35)
    
    # Align 12h EMA to 4h timeframe
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # need EMA, camarilla, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_4h[i]) or np.isnan(camarilla_r1_4h[i]) or 
            np.isnan(camarilla_s1_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 12h EMA34
        trend_up = close[i] > ema_34_12h_4h[i]
        trend_down = close[i] < ema_34_12h_4h[i]
        
        if position == 0:
            # Long entry: close above Camarilla R1 with volume and uptrend
            if (close[i] > camarilla_r1_4h[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below Camarilla S1 with volume and downtrend
            elif (close[i] < camarilla_s1_4h[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below Camarilla S1 or reverse signal
            if close[i] < camarilla_s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Camarilla R1 or reverse signal
            if close[i] > camarilla_r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0