#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
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
    
    # 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Previous day's OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: Range = High - Low
    range_1d = high_1d - low_1d
    r1 = close_1d + (range_1d * 1.0833)
    s1 = close_1d - (range_1d * 1.0833)
    
    # Use previous day's levels (shift by 1 to avoid look-ahead)
    r1_shifted = np.roll(r1, 1)
    s1_shifted = np.roll(s1, 1)
    r1_shifted[0] = np.nan
    s1_shifted[0] = np.nan
    
    # Align Camarilla levels to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_1d, r1_shifted)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1_shifted)
    
    # Volume spike filter: volume > 2.0x 24-period EMA (1 day of 1h bars)
    vol_ema24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_spike = volume > (2.0 * vol_ema24)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above 4h EMA50 (uptrend)
            if (price > r1_1h[i] and vol_spike[i] and price > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with volume spike and below 4h EMA50 (downtrend)
            elif (price < s1_1h[i] and vol_spike[i] and price < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below S1 (mean reversion to support)
            if price < s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above R1 (mean reversion to resistance)
            if price > r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals