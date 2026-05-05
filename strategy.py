#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation
# Camarilla levels provide intraday support/resistance that work in both bull/bear markets
# Breakouts above R1 or below S1 with volume confirmation indicate institutional interest
# 1d EMA50 filter ensures we only trade in the direction of the higher timeframe trend
# Volume spike (>1.5x 20-period MA) confirms conviction and reduces false breakouts
# Timeframe: 1h, HTF: 4h/1d. Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.
# Session filter: 08-20 UTC to trade only during active London/NY overlap and reduce noise.

name = "1h_Camarilla_R1S1_Breakout_1dEMA50_VolumeConfirm"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # need sufficient data for Camarilla calculation
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (using previous 4h bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels based on previous bar
    R1 = np.full(len(close_4h), np.nan)
    S1 = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        if np.isnan(high_4h[i-1]) or np.isnan(low_4h[i-1]) or np.isnan(close_4h[i-1]):
            continue
        diff = high_4h[i-1] - low_4h[i-1]
        R1[i] = close_4h[i-1] + (diff * 1.1 / 12)
        S1[i] = close_4h[i-1] - (diff * 1.1 / 12)
    
    # Align 4h Camarilla levels to 1h
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation on 1h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: close > R1 AND above 1d EMA50 AND volume spike
            if (close[i] > R1_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: close < S1 AND below 1d EMA50 AND volume spike
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: close < S1 (reversal to downside) OR below 1d EMA50
            if close[i] < S1_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: close > R1 (reversal to upside) OR above 1d EMA50
            if close[i] > R1_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals