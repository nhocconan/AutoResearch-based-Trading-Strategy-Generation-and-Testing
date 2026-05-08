#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_12hTrend_VolumeSpike"
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
    
    # Get 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h ATR(14) for volatility normalization
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1 = np.maximum(high_12h[1:] - low_12h[1:], 
                     np.maximum(np.abs(high_12h[1:] - close_12h[:-1]), 
                                np.abs(low_12h[1:] - close_12h[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])
    
    atr14 = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr14)
    
    # Calculate 4h Camarilla levels from previous 4h bar
    range_4h = high - low
    camarilla_r3 = close + range_4h * 1.1 / 4
    camarilla_s3 = close - range_4h * 1.1 / 4
    camarilla_r4 = close + range_4h * 1.1 / 2
    camarilla_s4 = close - range_4h * 1.1 / 2
    
    # Shift to get previous bar's levels (no look-ahead)
    camarilla_r3_prev = np.roll(camarilla_r3, 1)
    camarilla_s3_prev = np.roll(camarilla_s3, 1)
    camarilla_r4_prev = np.roll(camarilla_r4, 1)
    camarilla_s4_prev = np.roll(camarilla_s4, 1)
    camarilla_r3_prev[0] = np.nan
    camarilla_s3_prev[0] = np.nan
    camarilla_r4_prev[0] = np.nan
    camarilla_s4_prev[0] = np.nan
    
    # Volume spike detection: current volume > 2 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(atr14_12h_aligned[i]) or 
            np.isnan(camarilla_r3_prev[i]) or np.isnan(camarilla_s3_prev[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 with volume spike, above 12h EMA
            if (close[i] > camarilla_r3_prev[i] and vol_spike and 
                close[i] > ema_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume spike, below 12h EMA
            elif (close[i] < camarilla_s3_prev[i] and vol_spike and 
                  close[i] < ema_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 OR below 12h EMA
            if (close[i] < camarilla_s3_prev[i] or close[i] < ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 OR above 12h EMA
            if (close[i] > camarilla_r3_prev[i] or close[i] > ema_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Uses 4h Camarilla R3/S3 breakouts with volume confirmation and 12h EMA trend filter.
# - Enters long when price breaks above R3 (previous bar) with volume spike and above 12h EMA
# - Enters short when price breaks below S3 (previous bar) with volume spike and below 12h EMA
# - Exits when price breaks back below S3 (long) or above R3 (short) OR crosses 12h EMA
# - Volume spike filter ensures breakouts have conviction
# - 12h EMA filter ensures trading with higher timeframe trend
# - Camarilla levels provide natural support/resistance at key levels
# - Target: 80-160 total trades over 4 years (20-40/year) to minimize fee drag
# - Position size: 0.25 for balanced risk/return
# - Works in both bull and bear markets by following 12h trend direction
# - Volume confirmation reduces false breakouts in low-volume environments
# - Using 4h timeframe with 12h trend filter for better signal quality and reduced trade frequency
# - Based on top-performing patterns from DB showing 1.8+ Sharpe for similar configurations