# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
6h_1D_AroonOscillator_Volume_Breakout
Hypothesis: Use daily Aroon Oscillator to detect trend strength and direction, combined with volume confirmation and ATR filter for breakout entries.
Long when Aroon Oscillator > 50 (strong uptrend) and price breaks above 6h high of last 20 bars with volume > 1.8x average.
Short when Aroon Oscillator < -50 (strong downtrend) and price breaks below 6h low of last 20 bars with volume > 1.8x average.
Aroon Oscillator provides trend filter to avoid counter-trend trades, reducing whipsaw in choppy markets.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
Works in bull/bear via trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Aroon Oscillator (trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Aroon Oscillator calculation: Aroon Up - Aroon Down
    # Aroon Up = ((period - periods since highest high) / period) * 100
    # Aroon Down = ((period - periods since lowest low) / period) * 100
    period = 25  # typical Aroon period
    
    def calculate_aroon(high_arr, low_arr, period):
        n_days = len(high_arr)
        aroon_up = np.full(n_days, np.nan)
        aroon_down = np.full(n_days, np.nan)
        
        for i in range(period, n_days):
            # Find highest high in last 'period' days
            window_high = high_arr[i-period+1:i+1]
            highest_high_idx = np.argmax(window_high)
            periods_since_high = period - 1 - highest_high_idx
            aroon_up[i] = ((period - periods_since_high) / period) * 100
            
            # Find lowest low in last 'period' days
            window_low = low_arr[i-period+1:i+1]
            lowest_low_idx = np.argmin(window_low)
            periods_since_low = period - 1 - lowest_low_idx
            aroon_down[i] = ((period - periods_since_low) / period) * 100
        
        # For first 'period' days, use available data
        for i in range(period):
            window_high = high_arr[:i+1]
            highest_high_idx = np.argmax(window_high)
            periods_since_high = i - highest_high_idx
            aroon_up[i] = ((period - periods_since_high) / period) * 100
            
            window_low = low_arr[:i+1]
            lowest_low_idx = np.argmin(window_low)
            periods_since_low = i - lowest_low_idx
            aroon_down[i] = ((period - periods_since_low) / period) * 100
        
        aroon_osc = aroon_up - aroon_down
        return aroon_osc
    
    aroon_osc = calculate_aroon(high_1d, low_1d, period)
    
    # Precompute 6h breakout levels: high/low of last 20 periods
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: ATR(20) to avoid extreme volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]  # first period
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ma = pd.Series(atr_20).rolling(window=50, min_periods=50).mean().values
    
    # Aroon Oscillator alignment to 6h timeframe
    aroon_osc_aligned = align_htf_to_ltf(prices, df_1d, aroon_osc)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(aroon_osc_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_20[i]) or 
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Volatility filter: avoid extreme volatility
        vol_filter = atr_20[i] < atr_ma[i] * 2
        
        # Only trade during active session
        in_session = session_mask[i]
        
        if position == 0:
            # Long: strong uptrend (Aroon > 50) and breakout above recent high with volume
            if (aroon_osc_aligned[i] > 50 and close[i] > high_20[i] and 
                vol_confirm and vol_filter and in_session):
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend (Aroon < -50) and breakdown below recent low with volume
            elif (aroon_osc_aligned[i] < -50 and close[i] < low_20[i] and 
                  vol_confirm and vol_filter and in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend weakens (Aroon <= 0) or breakdown below recent low or outside session
            if (aroon_osc_aligned[i] <= 0 or close[i] < low_20[i] or 
                not vol_filter or not in_session):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend weakens (Aroon >= 0) or breakout above recent high or outside session
            if (aroon_osc_aligned[i] >= 0 or close[i] > high_20[i] or 
                not vol_filter or not in_session):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1D_AroonOscillator_Volume_Breakout"
timeframe = "6h"
leverage = 1.0