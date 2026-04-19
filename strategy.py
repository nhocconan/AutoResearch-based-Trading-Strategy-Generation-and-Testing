# 12h_1w_1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v1
# Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# Works in bull markets via breakouts above R1/S1, in bear via breakdowns
# 1w EMA200 filter ensures alignment with long-term trend
# Volume spike (>1.5x average) confirms breakout strength
# ATR-based stop limits downside
# Target: 20-30 trades/year to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w and 1d data for multi-timeframe analysis (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # 1w EMA200 for long-term trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 1d data for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d
    # R1 = close + (high - low) * 1.12
    # S1 = close - (high - low) * 1.12
    camarilla_range = high_1d - low_1d
    camarilla_r1 = close_1d + camarilla_range * 1.12
    camarilla_s1 = close_1d - camarilla_range * 1.12
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 12h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for 1w EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or \
           np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(atr_12h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_12h[i]
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        if position == 0:
            # Long: breakout above Camarilla R1 + volume + 1w uptrend
            if high[i] > camarilla_r1_aligned[i-1] and volume_filter and price > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Camarilla S1 + volume + 1w downtrend
            elif low[i] < camarilla_s1_aligned[i-1] and volume_filter and price < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below Camarilla S1 or ATR-based stop
            if close[i] < camarilla_s1_aligned[i] or close[i] < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above Camarilla R1 or ATR-based stop
            if close[i] > camarilla_r1_aligned[i] or close[i] > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals