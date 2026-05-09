#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume"
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
    
    # Get 4h data for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar
    # Camarilla uses previous bar's high, low, close
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Camarilla levels: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    camarilla_range = prev_high - prev_low
    camarilla_r1 = prev_close + camarilla_range * 1.1 / 12
    camarilla_s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1h
    camarilla_r1_1h = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_1h = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume average (20-period)
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_avg_1h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_1h[i]) or np.isnan(camarilla_s1_1h[i]) or 
            np.isnan(ema_50_1h[i]) or np.isnan(vol_avg_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_avg_1h[i]  # Volume confirmation
        in_session = 8 <= hours[i] <= 20
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with 4h uptrend and volume
            if (close[i] > camarilla_r1_1h[i] and 
                close[i] > ema_50_1h[i] and  # 4h uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S1 with 4h downtrend and volume
            elif (close[i] < camarilla_s1_1h[i] and 
                  close[i] < ema_50_1h[i] and  # 4h downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below Camarilla pivot (mean reversion)
            camarilla_pivot = (camarilla_r1_1h[i] + camarilla_s1_1h[i]) / 2
            if close[i] < camarilla_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above Camarilla pivot (mean reversion)
            camarilla_pivot = (camarilla_r1_1h[i] + camarilla_s1_1h[i]) / 2
            if close[i] > camarilla_pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals