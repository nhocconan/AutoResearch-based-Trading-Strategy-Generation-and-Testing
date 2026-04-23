#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike
- Long when price breaks above Camarilla R1 AND close > 1d EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below Camarilla S1 AND close < 1d EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses Camarilla midpoint (H5/L5) or reverses to opposite Camarilla level
- Uses 1d EMA34 for HTF trend alignment to avoid counter-trend entries
- Volume spike (2.0x) ensures institutional participation and reduces false breakouts
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (need high, low, close)
    # Camarilla: H5 = close + 1.1*(high-low)/2, L5 = close - 1.1*(high-low)/2
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # We use the previous completed 1d bar to calculate levels for current 4h bar
    prev_high = df_1d['high'].shift(1).values  # Previous 1d high
    prev_low = df_1d['low'].shift(1).values    # Previous 1d low
    prev_close = df_1d['close'].shift(1).values # Previous 1d close
    
    # Calculate Camarilla levels
    camarilla_high = prev_close + 1.1 * (prev_high - prev_low) / 2   # H5
    camarilla_low = prev_close - 1.1 * (prev_high - prev_low) / 2    # L5
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12    # R1
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12    # S1
    camarilla_mid = (camarilla_high + camarilla_low) / 2             # Midpoint (H5/L5 avg)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Volume confirmation: > 2.0x 20-period average (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for volume MA, 34 for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_r1_aligned[i]  # Break above R1
        breakout_down = close[i] < camarilla_s1_aligned[i]  # Break below S1
        
        # Trend filter (using 1d EMA34)
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation (strict: 2.0x average)
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Camarilla breakout up + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla breakout down + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midpoint OR reverses to S1
                if close[i] < camarilla_mid_aligned[i] or close[i] < camarilla_s1_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above midpoint OR reverses to R1
                if close[i] > camarilla_mid_aligned[i] or close[i] > camarilla_r1_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0