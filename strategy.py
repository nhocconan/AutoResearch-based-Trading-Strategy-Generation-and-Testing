#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day high, low, close for Camarilla levels
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Camarilla levels: R3, S3
    # R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 4h volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with volume and 1d uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > R3_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and 1d downtrend
            elif close[i] < S3_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S3 or volume drops
            if close[i] < S3_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R3 or volume drops
            if close[i] > R3_aligned[i] or volume[i] < vol_ma_20[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# - Camarilla R3/S3 levels act as strong support/resistance derived from prior day's range
# - Breakouts above R3/below S3 with volume indicate institutional participation
# - 1d EMA(34) ensures alignment with higher timeframe trend (avoid counter-trend trades)
# - Volume spike (1.8x average) filters weak breakouts
# - Works in bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend)
# - Position size 0.25 targets ~20-40 trades/year, avoiding fee drag
# - Exit at opposite S3/R3 level provides clear risk-reward in trending markets