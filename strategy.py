#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for each day
    range_ = prev_high - prev_low
    R3 = prev_close + range_ * 1.1 / 4
    R4 = prev_close + range_ * 1.1 / 2
    S3 = prev_close - range_ * 1.1 / 4
    S4 = prev_close - range_ * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 24-period average volume (3 days for 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(R3_4h[i]) or np.isnan(R4_4h[i]) or np.isnan(S3_4h[i]) or 
            np.isnan(S4_4h[i]) or np.isnan(ema_34_4h[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above R3 + above EMA34 + volume confirmation
            if (close[i] > R3_4h[i] and 
                close[i] > ema_34_4h[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + below EMA34 + volume confirmation
            elif (close[i] < S3_4h[i] and 
                  close[i] < ema_34_4h[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close drops below S3 or below EMA34
            if (close[i] < S3_4h[i] or close[i] < ema_34_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close rises above R3 or above EMA34
            if (close[i] > R3_4h[i] or close[i] > ema_34_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals