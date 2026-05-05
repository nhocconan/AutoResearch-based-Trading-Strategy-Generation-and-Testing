#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R3 AND 4h close > 4h EMA34 AND volume > 1.8x 20-period average
# Short when price breaks below Camarilla S3 AND 4h close < 4h EMA34 AND volume > 1.8x 20-period average
# Exit when price crosses 4h EMA34 (trend reversal)
# Uses 1h primary timeframe with 4h HTF for trend filter and 1d for Camarilla structure
# Discrete sizing (0.20) to limit fee drag and manage drawdown
# Target: 80-150 total trades over 4 years (20-38/year) for 1h timeframe
# Session filter: 08-20 UTC to reduce noise trades
# Works in both bull and bear markets by following the 4h trend while using 1h for entry timing

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 4h close for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from previous 1d bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 1h timeframe (use previous completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: volume > 1.8x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 4h close > 4h EMA34 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 4h close < 4h EMA34 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA34 (trend reversal)
            if close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA34 (trend reversal)
            if close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals