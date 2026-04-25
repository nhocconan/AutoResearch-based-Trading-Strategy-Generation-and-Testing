#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume spike confirmation.
Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag.
Uses discrete sizing (0.25) and tight volume threshold (2.0x) to avoid overtrading.
In uptrend (close > 1d EMA50): long R1 break, short S1 fade only with extreme volume.
In downtrend: short S1 break, long R1 fade only with extreme volume.
Exit on opposite Camarilla level touch or trend reversal. Works in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculations (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar (based on previous bar)
    R1_12h = np.full(len(close_12h), np.nan)
    S1_12h = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):
        # Camarilla levels based on previous 12h bar's range
        high_prev = high_12h[i-1]
        low_prev = low_12h[i-1]
        close_prev = close_12h[i-1]
        range_prev = high_prev - low_prev
        
        if range_prev > 0:
            R1_12h[i] = close_prev + (range_prev * 1.1 / 4)  # R1 level
            S1_12h[i] = close_prev - (range_prev * 1.1 / 4)  # S1 level
    
    # Align Camarilla levels to original timeframe
    R1_12h_aligned = align_htf_to_ltf(prices, df_12h, R1_12h)
    S1_12h_aligned = align_htf_to_ltf(prices, df_12h, S1_12h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (balanced to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R1_12h_aligned[i]) or np.isnan(S1_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime
                # Long: break above R1 with volume spike
                long_signal = (close[i] > R1_12h_aligned[i]) and vol_spike[i]
                # Short: break below S1 only if extreme volume spike (counter-trend fade)
                short_signal = (close[i] < S1_12h_aligned[i]) and vol_spike[i] and (volume[i] > (3.5 * vol_ma_20[i]))
            else:  # Downtrend regime
                # Short: break below S1 with volume spike
                short_signal = (close[i] < S1_12h_aligned[i]) and vol_spike[i]
                # Long: break above R1 only if extreme volume spike (counter-trend fade)
                long_signal = (close[i] > R1_12h_aligned[i]) and vol_spike[i] and (volume[i] > (3.5 * vol_ma_20[i]))
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: touch S1 or trend reversal
            exit_signal = (close[i] < S1_12h_aligned[i]) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: touch R1 or trend reversal
            exit_signal = (close[i] > R1_12h_aligned[i]) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0