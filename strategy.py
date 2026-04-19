#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d ADX trend filter and volume confirmation
# Williams Alligator identifies trends using smoothed medians (Jaws, Teeth, Lips)
# 1d ADX filters for strong trends (ADX > 25) to avoid choppy markets
# Volume confirmation ensures breakouts have participation
# Target: 75-200 total trades over 4 years (19-50/year) with disciplined entries
name = "4h_WilliamsAlligator_1dADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0.0
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0
            
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0
    
    # Smoothed averages
    def smooth_series(data, period):
        smoothed = np.zeros_like(data)
        if len(data) < period:
            return smoothed
        smoothed[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
        return smoothed
    
    atr_1d = smooth_series(tr_1d, 14)
    plus_di_1d = 100 * smooth_series(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * smooth_series(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    dx_1d[np.isnan(dx_1d) | np.isinf(dx_1d)] = 0
    adx_1d = smooth_series(dx_1d, 14)
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams Alligator on 4h (using median prices)
    median_price = (high + low) / 2
    
    # Smoothed medians with different periods
    def smma(data, period):
        smoothed = np.zeros_like(data)
        if len(data) < period:
            return smoothed
        smoothed[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
        return smoothed
    
    jaws = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)   # Red line
    lips = smma(median_price, 5)    # Green line
    
    # Align Alligator lines to 4h (they're already on 4h, but ensure alignment)
    # Actually, we calculate on 4h directly, so no alignment needed
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(jaws[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator signals: Lips > Teeth > Jaws = uptrend, Lips < Teeth < Jaws = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaws[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaws[i]
        
        if position == 0:
            # Long: Alligator uptrend + ADX > 25 (strong trend) + volume confirmation
            if (alligator_long and 
                adx_1d_aligned[i] > 25 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator downtrend + ADX > 25 (strong trend) + volume confirmation
            elif (alligator_short and 
                  adx_1d_aligned[i] > 25 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator turns downtrend or ADX weakens
            if (not alligator_long) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator turns uptrend or ADX weakens
            if (not alligator_short) or (adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals