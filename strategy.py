#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_4hTrend_1dVol_Trend_v1"
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
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate pivot and levels from previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's data
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # Set first value to NaN since no previous day exists
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    prev_daily_range = prev_high_1d - prev_low_1d
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1 = pivot + 1.1 * prev_daily_range / 6
    s1 = pivot - 1.1 * prev_daily_range / 6
    
    # Align Camarilla levels to 1h
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d volume average for volume trend filter (20-period)
    vol_avg_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 1h volume average for entry filter (20-period)
    vol_avg_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 8-20 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_avg_1h[i]) or np.isnan(vol_avg_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume conditions: 1h volume > 1.5 x 20-period average AND 1d volume > 1.2 x 20-period average
        vol_spike_1h = volume[i] > vol_avg_1h[i] * 1.5
        vol_trend_1d = df_1d['volume'].iloc[i // 24] > vol_avg_1d_aligned[i] * 1.2 if i // 24 < len(df_1d) else False
        
        if position == 0:
            # Long: Break above Camarilla R1 with uptrend (4h EMA50) and volume conditions
            if close[i] > r1_1h[i] and close[i] > ema50_4h_aligned[i] and vol_spike_1h and vol_trend_1d:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 with downtrend (4h EMA50) and volume conditions
            elif close[i] < s1_1h[i] and close[i] < ema50_4h_aligned[i] and vol_spike_1h and vol_trend_1d:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below Camarilla S1 OR trend turns down (4h EMA50)
            if close[i] < s1_1h[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price rises back above Camarilla R1 OR trend turns up (4h EMA50)
            if close[i] > r1_1h[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals