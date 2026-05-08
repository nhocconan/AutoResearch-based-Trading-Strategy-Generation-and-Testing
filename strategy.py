#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_Pivot_4hTrend_1dVolume_Signal"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_4h = close_4h + (high_4h - low_4h) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    s1_4h = close_4h - (high_4h - low_4h) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    r2_4h = close_4h + (high_4h - low_4h) * 1.1 / 6
    # S2 = C - (H - L) * 1.1 / 6
    s2_4h = close_4h - (high_4h - low_4h) * 1.1 / 6
    
    # Align Camarilla levels to 1h
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    r2_4h_aligned = align_htf_to_ltf(prices, df_4h, r2_4h)
    s2_4h_aligned = align_htf_to_ltf(prices, df_4h, s2_4h)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume ratio (current vs 20-period average)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / np.where(vol_ma_1d > 0, vol_ma_1d, 1.0)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or np.isnan(r2_4h_aligned[i]) or
            np.isnan(s2_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 and below S2 (mean reversion zone) + above daily EMA34 + volume expansion
            if (close[i] > s1_4h_aligned[i] and 
                close[i] < s2_4h_aligned[i] and
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio_1d_aligned[i] > 1.3):
                signals[i] = 0.20
                position = 1
            # Short: price below R1 and above R2 (mean reversion zone) + below daily EMA34 + volume expansion
            elif (close[i] < r1_4h_aligned[i] and 
                  close[i] > r2_4h_aligned[i] and
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio_1d_aligned[i] > 1.3):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below S1 or below daily EMA34
            if close[i] < s1_4h_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above R1 or above daily EMA34
            if close[i] > r1_4h_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals