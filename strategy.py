#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1w ADX trend filter and volume confirmation
# Williams Alligator identifies trends via SMAs of median price (Jaw/Teeth/Lips)
# 1w ADX > 25 filters for strong trending markets to avoid chop
# Volume confirmation ensures breakouts have participation
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
name = "6h_WilliamsAlligator_1wADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w ADX for trend strength filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate ADX on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0.0
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Plus and Minus Directional Movement
    plus_dm = np.zeros_like(high_1w)
    minus_dm = np.zeros_like(high_1w)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    for i in range(1, len(high_1w)):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0.0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0.0
    
    # Smoothed ATR, PlusDM, MinusDM
    def smooth_wilder(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = arr[:period].mean()
        for i in range(period, len(arr)):
            smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
        return smoothed
    
    atr_1w = smooth_wilder(tr_1w, 14)
    plus_dm_smooth = smooth_wilder(plus_dm, 14)
    minus_dm_smooth = smooth_wilder(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.zeros_like(atr_1w)
    minus_di = np.zeros_like(atr_1w)
    for i in range(len(atr_1w)):
        if atr_1w[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_1w[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_1w[i]
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
    
    # DX and ADX
    dx = np.zeros_like(atr_1w)
    for i in range(len(atr_1w)):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        else:
            dx[i] = 0.0
    
    adx_1w = smooth_wilder(dx, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Williams Alligator on 6h data
    # Median price = (High + Low) / 2
    median_price = (high + low) / 2
    
    # Jaw (13-period SMMA, 8 bars ahead)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    
    # Teeth (8-period SMMA, 5 bars ahead)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    
    # Lips (5-period SMMA, 3 bars ahead)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 20)  # Ensure enough data for Alligator and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + ADX > 25 + volume confirmation
            if (lips[i] > teeth[i] > jaw[i] and 
                adx_1w_aligned[i] > 25 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + ADX > 25 + volume confirmation
            elif (lips[i] < teeth[i] < jaw[i] and 
                  adx_1w_aligned[i] > 25 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Alligator alignment breaks or ADX < 20
            if not (lips[i] > teeth[i] > jaw[i]) or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Alligator alignment breaks or ADX < 20
            if not (lips[i] < teeth[i] < jaw[i]) or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals