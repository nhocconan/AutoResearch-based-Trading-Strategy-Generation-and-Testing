#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1h = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get daily data for Camarilla pivot and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC (for Camarilla calculation)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels R1 and S1 (inner bounds)
    camarilla_pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_range_1d = prev_high_1d - prev_low_1d
    camarilla_r1_1d = camarilla_pivot_1d + camarilla_range_1d * 1.1 / 12
    camarilla_s1_1d = camarilla_pivot_1d - camarilla_range_1d * 1.1 / 12
    
    # Align Camarilla levels to 1h
    camarilla_pivot_1h = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    camarilla_r1_1h = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1h = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Daily volume filter: above 2.0x 20-period average
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_1h[i]) or np.isnan(camarilla_s1_1h[i]) or 
            np.isnan(ema_20_1h[i]) or np.isnan(vol_ma_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = df_1d['volume'].iloc[i//24] > 2.0 * vol_ma_1d[i//24] if i//24 < len(df_1d) else False
        
        # Session filter: 08-20 UTC (reduce noise trades)
        in_session = 8 <= hours[i] <= 20
        
        if position == 0:
            # Long breakout: price breaks above camarilla R1 with 4h uptrend
            if (close[i] > camarilla_r1_1h[i] and 
                close[i] > ema_20_1h[i] and  # 4h uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short breakdown: price breaks below camarilla S1 with 4h downtrend
            elif (close[i] < camarilla_s1_1h[i] and 
                  close[i] < ema_20_1h[i] and  # 4h downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below camarilla pivot (mean reversion)
            if close[i] < camarilla_pivot_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above camarilla pivot (mean reversion)
            if close[i] > camarilla_pivot_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals