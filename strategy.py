#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d ATR(14) for volatility normalization
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])
    
    atr14 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Calculate 12h Camarilla levels from previous 12h bar
    range_12h = high - low
    camarilla_r1 = close + range_12h * 1.1 / 6
    camarilla_s1 = close - range_12h * 1.1 / 6
    
    # Shift to get previous bar's levels (no look-ahead)
    camarilla_r1_prev = np.roll(camarilla_r1, 1)
    camarilla_s1_prev = np.roll(camarilla_s1, 1)
    camarilla_r1_prev[0] = np.nan
    camarilla_s1_prev[0] = np.nan
    
    # Volume spike detection: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_r1_prev[i]) or 
            np.isnan(camarilla_s1_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R1 with volume spike, above 1d EMA
            if (close[i] > camarilla_r1_prev[i] and vol_spike and 
                close[i] > ema_val):
                signals[i] = 0.30
                position = 1
            # Enter short: price breaks below S1 with volume spike, below 1d EMA
            elif (close[i] < camarilla_s1_prev[i] and vol_spike and 
                  close[i] < ema_val):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 OR below 1d EMA
            if (close[i] < camarilla_s1_prev[i] or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above R1 OR above 1d EMA
            if (close[i] > camarilla_r1_prev[i] or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Uses 12h Camarilla R1/S1 breakouts with volume confirmation and 1d EMA trend filter.
# - Enters long when price breaks above R1 (previous bar) with volume spike and above 1d EMA
# - Enters short when price breaks below S1 (previous bar) with volume spike and below 1d EMA
# - Exits when price breaks back below S1 (long) or above R1 (short) OR crosses 1d EMA
# - Volume spike filter ensures breakouts have conviction
# - 1d EMA filter ensures trading with higher timeframe trend
# - Camarilla levels provide natural support/resistance at key levels
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# - Position size: 0.30 for balanced risk/return
# - Works in both bull and bear markets by following 1d trend direction
# - Volume confirmation reduces false breakouts in low-volume environments