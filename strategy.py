#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get volume data for confirmation
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d close for volatility filter
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ATR(14) on daily data
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0.0], tr])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_ma = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = atr_1d / np.where(atr_1d_ma > 0, atr_1d_ma, 1e-10)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if critical data is NaN
        if np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for current day
        day_idx = i // 96  # 96 4h bars per day (24h * 60min / 15min * 4)
        if day_idx >= len(df_1d):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        prev_close = close_1d[day_idx - 1] if day_idx > 0 else close_1d[0]
        prev_high = high_1d[day_idx - 1] if day_idx > 0 else high_1d[0]
        prev_low = low_1d[day_idx - 1] if day_idx > 0 else low_1d[0]
        
        # Camarilla R3 and S3 levels
        r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
        s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
        
        current_close = prices['close'].iloc[i]
        
        if position == 0:
            # Long: Price above R3 + 12h trend up + volume confirmation + volatility filter
            if (current_close > r3 and 
                ema_34_12h_aligned[i] > close_12h[min(i // 32, len(close_12h)-1)] and  # Simplified trend check
                volume[i] > vol_ma[i] * 1.5 and 
                vol_ratio_aligned[i] > 1.2):
                signals[i] = 0.25
                position = 1
            # Short: Price below S3 + 12h trend down + volume confirmation + volatility filter
            elif (current_close < s3 and 
                  ema_34_12h_aligned[i] < close_12h[min(i // 32, len(close_12h)-1)] and
                  volume[i] > vol_ma[i] * 1.5 and 
                  vol_ratio_aligned[i] > 1.2):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price below S3 or trend change
            if current_close < s3 or ema_34_12h_aligned[i] < close_12h[min(i // 32, len(close_12h)-1)]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price above R3 or trend change
            if current_close > r3 or ema_34_12h_aligned[i] > close_12h[min(i // 32, len(close_12h)-1)]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals