#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h trend filter (EMA34)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = pd.Series(df_4h['close'].values)
    ema34_4h = close_4h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate 1d Camarilla levels (R3, S3) for structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's Camarilla levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla R3 and S3 levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align to 1h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: current volume > 1.8x 24-period average
    vol_series = pd.Series(volume)
    vol_ma24 = vol_series.rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 24)  # warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        vol_ok = volume[i] > 1.8 * vol_ma24[i]
        
        if position == 0:
            if in_session and vol_ok:
                # Long: Price breaks above R3 with 4h uptrend
                if close[i] > R3_aligned[i] and close[i] > ema34_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: Price breaks below S3 with 4h downtrend
                elif close[i] < S3_aligned[i] and close[i] < ema34_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Exit long: Price crosses back below S3 (mean reversion)
            if close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price crosses back above R3
            if close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals