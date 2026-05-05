#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume confirmation and 1d EMA34 trend filter
# Long when price breaks above 12h Camarilla R3 level AND 1d volume > 1.5x 20-period average AND close > 1d EMA34
# Short when price breaks below 12h Camarilla S3 level AND 1d volume > 1.5x 20-period average AND close < 1d EMA34
# Exit when price crosses 12h Camarilla midpoint (mean reversion)
# Uses 12h primary timeframe with 1d HTF for volume confirmation and trend filter
# Camarilla levels provide institutional support/resistance; volume confirms conviction; EMA filter avoids counter-trend trades
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_1dVolume_1dEMA34"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for volume confirmation and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d volume spike filter
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_filter_1d = vol_1d > (1.5 * vol_ma_20)
    else:
        volume_filter_1d = np.zeros(len(df_1d), dtype=bool)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    if len(close_1d) >= 34:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_34_1d = np.full(len(close_1d), np.nan)
    
    # Get 12h data ONCE before loop for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R3, S3, midpoint)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True range approximation for Camarilla (using daily range)
    range_12h = high_12h - low_12h
    camarilla_r3 = close_12h + (range_12h * 1.1 / 2)
    camarilla_s3 = close_12h - (range_12h * 1.1 / 2)
    camarilla_mid = (camarilla_r3 + camarilla_s3) / 2
    
    # Align 1d indicators to 12h timeframe
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Align 12h indicators to 12h timeframe (same df_12h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_12h, camarilla_mid)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND volume spike AND above 1d EMA34
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_filter_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND volume spike AND below 1d EMA34
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_filter_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla midpoint (mean reversion)
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla midpoint (mean reversion)
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals