#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h trend following with 1d Camarilla pivot breakout.
    # Long when price breaks above 1d R3 with volume confirmation and 12h EMA21 uptrend.
    # Short when price breaks below 1d S3 with volume confirmation and 12h EMA21 downtrend.
    # Uses 12h close for entry, avoids false breakouts in ranging markets.
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r3_1d = pivot_1d + range_1d * 1.1
    s3_1d = pivot_1d - range_1d * 1.1
    
    # Align HTF pivot levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate EMA21 on 12h for trend filter
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or np.isnan(ema21[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: EMA21 slope (uptrend if close > ema21, downtrend if close < ema21)
        uptrend = close[i] > ema21[i]
        downtrend = close[i] < ema21[i]
        
        # Breakout conditions with volume confirmation
        long_breakout = (high[i] > r3_1d_aligned[i]) and volume_confirm[i]
        short_breakout = (low[i] < s3_1d_aligned[i]) and volume_confirm[i]
        
        # Exit conditions: price returns to 1d pivot
        long_exit = close[i] <= pivot_1d_aligned[i]
        short_exit = close[i] >= pivot_1d_aligned[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and uptrend and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and downtrend and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_ema_volume_v1"
timeframe = "12h"
leverage = 1.0