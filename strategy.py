#!/usr/bin/env python3
"""
1h_TRIX_Trend_VolumeSpike_Conservative
Hypothesis: Use TRIX (15-period) for momentum with volume spike confirmation (>2x average) and 1d trend filter (price above/below 200 EMA). Conservative sizing (0.20) and session filter (08-20 UTC) to limit trades to 15-30/year. TRIX captures momentum shifts in both bull and bear markets, volume confirms institutional participation, and daily trend filter avoids counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate TRIX (15-period) on 1h close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 / np.roll(ema3, 1)) - 1  # Percentage change
    trix[0] = 0  # First value has no previous
    
    # Volume confirmation: >2x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200 and TRIX to stabilize
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(trix[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA200
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # TRIX momentum signals
        trix_bullish = trix[i] > 0 and trix[i] > trix[i-1]  # Rising and positive
        trix_bearish = trix[i] < 0 and trix[i] < trix[i-1]  # Falling and negative
        
        # Entry conditions
        long_entry = trix_bullish and vol_confirm and uptrend
        short_entry = trix_bearish and vol_confirm and downtrend
        
        # Exit conditions: TRIX crosses zero or momentum fades
        long_exit = trix[i] <= 0 or (trix[i] < trix[i-1] and trix[i] > 0)  # Peak in positive territory
        short_exit = trix[i] >= 0 or (trix[i] > trix[i-1] and trix[i] < 0)  # Trough in negative territory
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_TRIX_Trend_VolumeSpike_Conservative"
timeframe = "1h"
leverage = 1.0