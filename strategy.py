#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    """
    12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
    - Long: Price breaks above Camarilla R3 with 1d EMA34 trend and volume spike
    - Short: Price breaks below Camarilla S3 with 1d EMA34 trend and volume spike
    - Exit: Price crosses back through Camarilla pivot (P)
    - Volume spike: current volume > 2.0 x 24-period average
    - Target: 12-37 trades/year on 12h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels from prior 12h candle
    # Camarilla uses previous period's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point
    P = (prev_high + prev_low + prev_close) / 3
    # Camarilla levels
    range_val = prev_high - prev_low
    R3 = P + range_val * 1.1 / 2
    S3 = P - range_val * 1.1 / 2
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34[i] = close_1d[i] * (2 / (34 + 1)) + ema_34[i-1] * (1 - 2 / (34 + 1))
    
    # Align EMA34 to 12h
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike detection (24-period for 12h)
    vol_avg = np.full(n, np.nan)
    for i in range(24, n):
        vol_avg[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24)  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(P[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema_34_12h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 24-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        # Trend filter: price > EMA34 for long, price < EMA34 for short
        uptrend = close[i] > ema_34_12h[i]
        downtrend = close[i] < ema_34_12h[i]
        
        if position == 0:
            # Long: Break above R3 with uptrend and volume spike
            if (close[i] > R3[i] and uptrend and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with downtrend and volume spike
            elif (close[i] < S3[i] and downtrend and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below pivot P
            if close[i] < P[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above pivot P
            if close[i] > P[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals