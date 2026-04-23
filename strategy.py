#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
- Camarilla R3/S3 levels provide institutional support/resistance on 1h
- 4h EMA(34) ensures alignment with higher timeframe trend to reduce counter-trend trades
- Volume spike (>2.0x 20-period average) confirms strong participation
- Session filter (08-20 UTC) reduces noise trades
- Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag
- Works in both bull and bear markets by trading with the 4h trend when price breaks Camarilla levels with volume
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(34) for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h EMA(20) for Camarilla base (previous bar's close)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # EMA4h, EMA20, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using previous bar's OHLC
        # Camarilla uses previous day's range, but we'll use previous bar for 1h timeframe
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        
        # Camarilla R3 and S3 levels
        rang = prev_high - prev_low
        r3 = prev_close + rang * 1.1/4
        s3 = prev_close - rang * 1.1/4
        
        # Breakout signals with trend filter and volume confirmation
        # Long: price breaks above R3 + uptrend + volume spike
        # Short: price breaks below S3 + downtrend + volume spike
        long_signal = (close[i] > r3 and 
                      close[i] > ema_34_4h_aligned[i] and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < s3 and 
                       close[i] < ema_34_4h_aligned[i] and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: price returns to EMA20 or opposite Camarilla level
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns below EMA20 or breaks below S3
                if (close[i] < ema_20[i] or 
                    close[i] < s3):
                    exit_signal = True
            elif position == -1:
                # Exit short: price returns above EMA20 or breaks above R3
                if (close[i] > ema_20[i] or 
                    close[i] > r3):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Trend_Volume_SessionFilter"
timeframe = "1h"
leverage = 1.0