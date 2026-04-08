#!/usr/bin/env python3
# 12h_keltner_breakout_1d_trend_volume_v1
# Hypothesis: Keltner Channel breakout with 1d EMA trend filter and volume confirmation.
# Enter long when price breaks above upper Keltner (EMA + 2*ATR) with volume > 1.5x average and price above 1d EMA(50).
# Enter short when price breaks below lower Keltner (EMA - 2*ATR) with volume > 1.5x average and price below 1d EMA(50).
# Exit on opposite signal or when price crosses back through the EMA middle line.
# Designed for 12h timeframe to capture multi-day trends while avoiding whipsaws.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_keltner_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Keltner Channel components on 12h
    # Middle line: EMA(20)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    # ATR(10) for channel width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    # Upper and lower bands
    upper_keltner = ema_20 + 2 * atr
    lower_keltner = ema_20 - 2 * atr
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below EMA middle or opposite signal with volume
            ema_cross = close[i] < ema_20[i]
            opposite_signal = (close[i] < lower_keltner[i] and 
                             volume[i] > 1.5 * avg_volume[i] and 
                             close[i] < ema_50_1d_aligned[i])
            if ema_cross or opposite_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above EMA middle or opposite signal with volume
            ema_cross = close[i] > ema_20[i]
            opposite_signal = (close[i] > upper_keltner[i] and 
                             volume[i] > 1.5 * avg_volume[i] and 
                             close[i] > ema_50_1d_aligned[i])
            if ema_cross or opposite_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above upper Keltner with volume and 1d uptrend
            if close[i] > upper_keltner[i] and volume_ok and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Keltner with volume and 1d downtrend
            elif close[i] < lower_keltner[i] and volume_ok and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals