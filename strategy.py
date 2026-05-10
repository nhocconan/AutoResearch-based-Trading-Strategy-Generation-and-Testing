#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Trade breakouts from Camarilla R3/S3 levels (from 1d) in direction of 1d trend (EMA34) with volume spike (>1.5x average).
# Works in bull/bear markets by following daily trend; avoids false breakouts with volume confirmation.
# Target: 50-150 trades over 4 years (~12-37/year).

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Get 1d data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla calculations
    R3 = np.concatenate([[np.nan], prev_close + 1.1 * (prev_high - prev_low) / 6])
    S3 = np.concatenate([[np.nan], prev_close - 1.1 * (prev_high - prev_low) / 6])
    R4 = np.concatenate([[np.nan], prev_close + 1.1 * (prev_high - prev_low) / 2])
    S4 = np.concatenate([[np.nan], prev_close - 1.1 * (prev_high - prev_low) / 2])
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 6h
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R4_6h = align_htf_to_ltf(prices, df_1d, R4)
    S4_6h = align_htf_to_ltf(prices, df_1d, S4)
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or np.isnan(ema_34_1d_6h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 in uptrend with volume
            if (close[i] > ema_34_1d_6h[i] and  # 1d uptrend
                close[i] > R3_6h[i] and 
                close[i-1] <= R3_6h[i-1] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in downtrend with volume
            elif (close[i] < ema_34_1d_6h[i] and  # 1d downtrend
                  close[i] < S3_6h[i] and 
                  close[i-1] >= S3_6h[i-1] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below S3 or stoploss
            if close[i] < S3_6h[i] or (i > 0 and low[i] < R3_6h[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above R3 or stoploss
            if close[i] > R3_6h[i] or (i > 0 and high[i] > S3_6h[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals