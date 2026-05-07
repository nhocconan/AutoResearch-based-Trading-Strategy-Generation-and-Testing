#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Load daily data ONCE for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    close_prev = df_1d['close'].shift(1).values
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    
    # Camarilla R3, R4, S3, S4
    R4 = close_prev + (high_prev - low_prev) * 1.1 / 2
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    S4 = close_prev - (high_prev - low_prev) * 1.1 / 2
    
    # Align daily levels to 4h
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # enough for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R3_4h[i]) or np.isnan(R4_4h[i]) or np.isnan(S3_4h[i]) or 
            np.isnan(S4_4h[i]) or np.isnan(ema_34_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above R3 with volume in daily uptrend
            if close[i] > R3_4h[i] and vol_condition and ema_34_4h[i] > ema_34_4h[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume in daily downtrend
            elif close[i] < S3_4h[i] and vol_condition and ema_34_4h[i] < ema_34_4h[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close back below R3 or trend reversal
            if close[i] < R3_4h[i] or ema_34_4h[i] < ema_34_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close back above S3 or trend reversal
            if close[i] > S3_4h[i] or ema_34_4h[i] > ema_34_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with daily trend filter and volume confirmation
# - Uses Camarilla levels from previous day (R3, S3) as key support/resistance
# - Long when price breaks above R3 with volume spike in daily uptrend (EMA34 rising)
# - Short when price breaks below S3 with volume spike in daily downtrend (EMA34 falling)
# - Volume confirmation (1.5x 20-period MA) reduces false breakouts
# - Exits when price returns to the broken level or trend reverses
# - Position size 0.25 limits risk and keeps trade frequency ~20-50/year
# - Works in bull markets (buying R3 breaks in uptrend) and bear markets (selling S3 breaks in downtrend)
# - Daily trend filter ensures alignment with higher timeframe, reducing whipsaws
# - Target: 80-150 total trades over 4 years (20-38/year) to avoid fee drag
# - Camarilla levels are proven to work well on crypto, especially with volume confirmation