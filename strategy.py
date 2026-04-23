#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation
- Long when price breaks above 4h Camarilla R1 AND price > 12h EMA50 AND volume > 2.0x 20-period average
- Short when price breaks below 4h Camarilla S1 AND price < 12h EMA50 AND volume > 2.0x 20-period average
- Exit when price crosses the 4h Camarilla midpoint (mean reversion to median)
- Uses 12h EMA50 for HTF trend alignment to avoid counter-trend entries
- Volume spike ensures institutional participation and reduces false breakouts
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
    
    # Get 12h data for EMA50 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 4h data for Camarilla pivot levels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (based on previous bar's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using previous 4h bar's OHLC to avoid look-ahead
    prev_high = df_4h['high'].shift(1)
    prev_low = df_4h['low'].shift(1)
    prev_close = df_4h['close'].shift(1)
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_mid = (camarilla_r1 + camarilla_s1) / 2.0
    
    # Convert to numpy arrays and align to 4h timeframe
    camarilla_r1_vals = camarilla_r1.values
    camarilla_s1_vals = camarilla_s1.values
    camarilla_mid_vals = camarilla_mid.values
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_vals)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_vals)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_4h, camarilla_mid_vals)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 51, 21)  # Need 20 for volume MA, 51 for EMA50 (50+1), 1 for Camarilla (shifted)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions (using 4h Camarilla levels)
        breakout_up = close[i] > camarilla_r1_aligned[i]  # Break above Camarilla R1
        breakout_down = close[i] < camarilla_s1_aligned[i]  # Break below Camarilla S1
        
        # Trend filter (using 12h EMA50)
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses 4h Camarilla midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below midpoint
                if close[i] < camarilla_mid_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above midpoint
                if close[i] > camarilla_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0