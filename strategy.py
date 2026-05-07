# Hypothesis: 4h/12h Camarilla Pivot Reversal with Volume Spike and ADX Trend Filter
# - Uses 12h Camarilla levels for higher timeframe structure
# - 4h price action at these levels with volume confirmation for entry
# - ADX > 25 filters for trending conditions to avoid false breakouts in ranges
# - Works in bull markets (buy dips to S1/S2 in uptrend) and bear markets (sell rallies to R1/R2 in downtrend)
# - Target: 20-40 trades/year to avoid fee drag while capturing meaningful moves

#!/usr/bin/env python3
name = "4h_12h_Camarilla_Pivot_Reversal_ADX_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels from previous 12h bar
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels (using widely accepted multipliers)
    s1 = prev_close - (range_hl * 1.08 / 2)
    s2 = prev_close - (range_hl * 1.16 / 2)
    r1 = prev_close + (range_hl * 1.08 / 2)
    r2 = prev_close + (range_hl * 1.16 / 2)
    
    # Align 12h levels to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    
    # ADX trend filter on 12h data (ADX > 25 = trending)
    # Calculate True Range
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Directional Movement
    up_move = df_12h['high'] - df_12h['high'].shift(1)
    down_move = df_12h['low'].shift(1) - df_12h['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(span=14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean() / tr_ma
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean() / tr_ma
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=14, adjust=False).mean()
    adx_values = adx.values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_values)
    
    # Volume spike detection: 4-period average (2 periods of 12h = 1 day)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 4)  # Wait for ADX and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near S1/S2 with volume spike and uptrend (ADX rising)
            vol_condition = volume[i] > vol_ma_4[i] * 1.5
            uptrend = adx_aligned[i] > 25 and adx_aligned[i] > adx_aligned[i-1]
            
            # Enter long near support levels in uptrend
            if (close[i] <= s1_aligned[i] * 1.005 or close[i] <= s2_aligned[i] * 1.005) and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Enter short near resistance levels in downtrend
            elif (close[i] >= r1_aligned[i] * 0.995 or close[i] >= r2_aligned[i] * 0.995) and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches middle or opposite resistance, or trend weakens
            if (close[i] >= (pivot := (s1_aligned[i] + r1_aligned[i]) / 2) * 0.995 or 
                adx_aligned[i] < 20):  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches middle or opposite support, or trend weakens
            if (close[i] <= (pivot := (s1_aligned[i] + r1_aligned[i]) / 2) * 1.005 or 
                adx_aligned[i] < 20):  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals