#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hTrend_1dVolatilityFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous 4h bar's OHLC for Camarilla calculation (using 4h close)
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    prev_close_4h = df_4h['close'].shift(1).values
    
    # Calculate Camarilla levels: R3, S3 (4h)
    camarilla_r3 = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 2
    camarilla_s3 = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # 4h EMA trend filter (21-period)
    ema_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ATR for volatility filter (14-period)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 1h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Wait for warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 4h uptrend and low volatility
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_4h_aligned[i] and 
                atr_1d_aligned[i] < np.nanmedian(atr_1d_aligned[max(0, i-100):i+1])):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with 4h downtrend and low volatility
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_4h_aligned[i] and 
                  atr_1d_aligned[i] < np.nanmedian(atr_1d_aligned[max(0, i-100):i+1])):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price crosses below R3 or drops below 4h EMA
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price crosses above S3 or rises above 4h EMA
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals