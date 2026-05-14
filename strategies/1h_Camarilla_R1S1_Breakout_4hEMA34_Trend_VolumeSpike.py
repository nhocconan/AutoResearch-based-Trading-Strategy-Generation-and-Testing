#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike confirmation
- Long when price breaks above 1h Camarilla R1 AND price > 4h EMA34 AND volume > 2.0x 20-period average
- Short when price breaks below 1h Camarilla S1 AND price < 4h EMA34 AND volume > 2.0x 20-period average
- Exit when price crosses the 1h Camarilla pivot point (mean reversion to median)
- Uses 4h EMA34 for trend alignment to avoid counter-trend trades and capture major trend
- Volume spike ensures institutional participation and reduces false breakouts
- Uses 1h primary timeframe with 4h HTF for signal direction to balance trade frequency
- Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag
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
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate 4h EMA34
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Get 1h data for Camarilla levels
    # Camarilla formulas: Pivot = (H+L+C)/3, Range = H-L
    # R1 = Pivot + Range * 1.1/12, S1 = Pivot - Range * 1.1/12
    typical_price = (high + low + close) / 3.0
    pivot = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    range_hl = pd.Series(high - low).rolling(window=20, min_periods=20).mean().values
    camarilla_r1 = pivot + range_hl * 1.1 / 12.0
    camarilla_s1 = pivot - range_hl * 1.1 / 12.0
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 35, 21)  # Need 20 for Camarilla, 35 for EMA34 (34+1), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or 
            np.isnan(pivot[i]) or 
            np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_r1[i]  # Break above Camarilla R1
        breakout_down = close[i] < camarilla_s1[i]  # Break below Camarilla S1
        
        # Trend filter
        uptrend = close[i] > ema34_4h_aligned[i]
        downtrend = close[i] < ema34_4h_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: bearish breakout + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price crosses Camarilla pivot point (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below pivot
                if close[i] < pivot[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above pivot
                if close[i] > pivot[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0