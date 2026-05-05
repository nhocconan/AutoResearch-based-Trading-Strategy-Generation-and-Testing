#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 Breakout with 4h EMA34 Trend Filter and Volume Spike
# Long when price breaks above Camarilla R3 level AND price > 4h EMA34 (uptrend) AND volume spike (2.0x 20-bar MA)
# Short when price breaks below Camarilla S3 level AND price < 4h EMA34 (downtrend) AND volume spike
# Camarilla levels from 1d OHLC provide institutional support/resistance; 4h EMA34 filters trend alignment
# Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe to avoid fee drag.
# Uses discrete position size 0.20 to balance return and drawdown.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Trend_VolumeSpike_Session"
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
    
    # Session filter: 08-20 UTC (pre-compute hours index)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d OHLC
    # R3 = close + 1.1*(high - low)/4
    # S3 = close - 1.1*(high - low)/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Get 4h data ONCE before loop for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation on 1h (threshold: 2.0x 20-bar MA)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND price > 4h EMA34 (uptrend) AND volume spike AND in session
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume_spike[i] and 
                in_session):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 AND price < 4h EMA34 (downtrend) AND volume spike AND in session
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_spike[i] and 
                  in_session):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 OR price < 4h EMA34 (trend break)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 OR price > 4h EMA34 (trend break)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals