#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_camarilla_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend and volatility filters
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels using previous day's data
    pivot_point = (high_1d + low_1d + close_1d) / 3.0
    daily_range = high_1d - low_1d
    
    # Camarilla levels (using previous day's OHLC)
    r4 = close_1d + daily_range * 1.1 / 2
    r3 = close_1d + daily_range * 1.1 / 4
    r2 = close_1d + daily_range * 1.1 / 6
    r1 = close_1d + daily_range * 1.1 / 12
    s1 = close_1d - daily_range * 1.1 / 12
    s2 = close_1d - daily_range * 1.1 / 6
    s3 = close_1d - daily_range * 1.1 / 4
    s4 = close_1d - daily_range * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # ATR for volatility filter (14-period on 4h)
    tr1 = pd.Series(high_4h).subtract(pd.Series(low_4h)).abs()
    tr2 = pd.Series(high_4h).subtract(pd.Series(close_4h).shift(1)).abs()
    tr3 = pd.Series(low_4h).subtract(pd.Series(close_4h).shift(1)).abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    atr_4h_ma = pd.Series(atr_4h).rolling(window=20, min_periods=20).mean().values
    vol_filter_4h = atr_4h > atr_4h_ma
    
    # Volume spike: current volume > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(vol_filter_4h[i]) or
            np.isnan(session_filter[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 or trend reverses
            if close[i] < s3_aligned[i] or close[i] < ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 or trend reverses
            if close[i] > r3_aligned[i] or close[i] > ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Trend filter: price vs 4h EMA20
            uptrend = close[i] > ema_20_4h_aligned[i]
            downtrend = close[i] < ema_20_4h_aligned[i]
            
            # Long: price breaks above R3 + uptrend + volume spike + vol filter
            if (close[i] > r3_aligned[i] and 
                uptrend and 
                vol_spike[i] and
                vol_filter_4h[i]):
                position = 1
                signals[i] = 0.20
            # Short: price breaks below S3 + downtrend + volume spike + vol filter
            elif (close[i] < s3_aligned[i] and 
                  downtrend and 
                  vol_spike[i] and
                  vol_filter_4h[i]):
                position = -1
                signals[i] = -0.20
    
    return signals