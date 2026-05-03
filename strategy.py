#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA34 trend filter and volume spike
# Camarilla pivots identify key intraday support/resistance levels. Breakouts above R3 or below S3
# with volume confirmation indicate strong momentum. 12h EMA34 filter ensures trades align with
# higher timeframe trend to avoid false breakouts. Designed for 12-30 trades/year on 6h to
# minimize fee drag while capturing trending moves in both bull and bear markets.

name = "6h_Camarilla_R3_S3_Breakout_12hEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have previous bar for pivot calc
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_12h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using previous 12h bar (completed bar)
        idx_12h = i // 48  # 6h bars per 12h bar (12h/6h = 2, but we use completed bar so look back)
        if idx_12h < 1:
            continue
            
        # Get completed 12h bar (idx_12h-1) for pivot calculation
        prev_12h_idx = idx_12h - 1
        if prev_12h_idx < 0 or prev_12h_idx >= len(df_12h):
            continue
            
        high_12h = df_12h['high'].iloc[prev_12h_idx]
        low_12h = df_12h['low'].iloc[prev_12h_idx]
        close_12h = df_12h['close'].iloc[prev_12h_idx]
        
        # Camarilla pivot levels
        pivot = (high_12h + low_12h + close_12h) / 3
        range_12h = high_12h - low_12h
        r3 = pivot + (range_12h * 1.1 / 4)
        s3 = pivot - (range_12h * 1.1 / 4)
        r4 = pivot + (range_12h * 1.1 / 2)
        s4 = pivot - (range_12h * 1.1 / 2)
        
        # Volume confirmation: 20-period EMA
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        if position == 0:
            # Long: break above R3 with volume spike and 12h uptrend
            if close[i] > r3 and volume_spike and ema_34_12h_aligned[i] > close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume spike and 12h downtrend
            elif close[i] < s3 and volume_spike and ema_34_12h_aligned[i] < close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below pivot or loses 12h uptrend
            if close[i] < pivot or ema_34_12h_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above pivot or loses 12h downtrend
            if close[i] > pivot or ema_34_12h_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals