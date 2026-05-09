#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Hourly OHLC from previous hour (Camarilla calculation)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(prices['open'].values, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    prev_open[0] = prices['open'].values[0]
    
    # Calculate Camarilla levels (R3, S3)
    range_ = prev_high - prev_low
    close_prev = prev_close
    
    r3 = close_prev + range_ * 1.1 / 4
    s3 = close_prev - range_ * 1.1 / 4
    
    # 4h trend: EMA50 on 4h
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d trend filter: EMA100 on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    ema100_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # Volume filter: volume > 1.4x 24-period SMA
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > 1.4 * vol_ma24
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(r3[i]) or np.isnan(s3[i]) or \
           np.isnan(ema50_4h_aligned[i]) or np.isnan(ema100_1d_aligned[i]) or \
           np.isnan(vol_ma24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: breakout above R3 with 4h/1d uptrend and volume
            if (price > r3[i] and 
                price > ema50_4h_aligned[i] and 
                price > ema100_1d_aligned[i] and 
                vol_filter[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
                continue
            
            # Short: breakdown below S3 with 4h/1d downtrend and volume
            elif (price < s3[i] and 
                  price < ema50_4h_aligned[i] and 
                  price < ema100_1d_aligned[i] and 
                  vol_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to 4h EMA or loses volume/filter
            if (price < ema50_4h_aligned[i] or 
                not vol_filter[i] or 
                not session_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to 4h EMA or loses volume/filter
            if (price > ema50_4h_aligned[i] or 
                not vol_filter[i] or 
                not session_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals