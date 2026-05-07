#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Price breaking above Camarilla R1 or below S1 with 12h EMA50 trend filter and volume confirmation captures high-probability breakout moves. Camarilla levels provide statistically significant support/resistance, 12h trend ensures alignment with higher timeframe momentum, and volume filter adds confirmation. Designed for 4h timeframe to limit trade frequency and reduce fee drag, targeting 20-50 trades/year per symbol.
"""
name = "4h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla levels from previous day (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    camarilla_R1 = np.full(len(df_1d), np.nan)
    camarilla_S1 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        # Previous day's high, low, close
        ph = df_1d['high'].iloc[i-1]
        pl = df_1d['low'].iloc[i-1]
        pc = df_1d['close'].iloc[i-1]
        
        # Camarilla formulas
        camarilla_R1[i] = pc + (ph - pl) * 1.1 / 12
        camarilla_S1[i] = pc - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1, additional_delay_bars=0)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1, additional_delay_bars=0)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Start after first day to have Camarilla levels
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 12h uptrend + volume
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + 12h downtrend + volume
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to the opposite Camarilla level (mean reversion)
            if position == 1:
                if close[i] < camarilla_S1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > camarilla_R1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals