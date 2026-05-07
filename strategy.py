#!/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "1h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for previous day
    # Camarilla uses previous day's high, low, close
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Calculate R3 and S3 levels
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Align Camarilla levels to 1h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 1h volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA and Camarilla calculation
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with volume and 1d uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]  # Rising EMA
            
            if close[i] > R3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: break below S3 with volume and 1d downtrend
            elif close[i] < S3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below S3 or volume drops
            if close[i] < S3_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above R3 or volume drops
            if close[i] > R3_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R3/S3 breakout with 1d EMA trend filter and volume confirmation
# - Camarilla R3/S3 levels identify key resistance/support for intraday breakouts
# - 1d EMA(34) ensures alignment with daily trend (works in bull/bear markets)
# - Volume spike (1.8x average) confirms institutional participation
# - Position size 0.20 targets 15-30 trades/year, avoiding fee drag
# - Exit at opposite Camarilla level provides clear risk/reward
# - Works in bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend)