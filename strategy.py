#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Camarilla pivot levels (R3/S3) act as strong support/resistance; breakouts with volume
# indicate institutional participation. 12h EMA50 ensures trend alignment to avoid counter-trend trades.
# Volume confirmation (2.0x 20-period EMA) filters weak breakouts. Designed for 4h timeframe
# to target 20-50 trades/year (75-200 total over 4 years) with discrete sizing (0.25).
# Works in bull markets by buying breakouts in uptrends and in bear markets by selling
# breakdowns in downtrends, avoiding range-bound whipsaws.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (using prior 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels for each 4h bar using prior 1d bar's OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        # Need prior 1d bar data (1d bar must be closed)
        if i < 96:  # 96*15m = 24h, need at least one full 1d bar before current 4h bar
            continue
            
        # Get index of prior 1d bar in 1d dataframe
        # 1d bar index = floor(i / 96) - 1 (since we want prior completed 1d bar)
        idx_1d = (i // 96) - 1
        if idx_1d < 0 or idx_1d >= len(df_1d):
            continue
            
        # Calculate Camarilla levels from prior 1d bar
        h_1d = df_1d['high'].iloc[idx_1d]
        l_1d = df_1d['low'].iloc[idx_1d]
        c_1d = df_1d['close'].iloc[idx_1d]
        
        camarilla_r3[i] = c_1d + (h_1d - l_1d) * 1.1 / 4
        camarilla_s3[i] = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # Volume confirmation: 2.0x 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + price above 12h EMA50 (uptrend)
            if (close[i] > camarilla_r3[i] and volume_spike and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume spike + price below 12h EMA50 (downtrend)
            elif (close[i] < camarilla_s3[i] and volume_spike and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 OR price below 12h EMA50 (trend change)
            if close[i] < camarilla_s3[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 OR price above 12h EMA50 (trend change)
            if close[i] > camarilla_r3[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals