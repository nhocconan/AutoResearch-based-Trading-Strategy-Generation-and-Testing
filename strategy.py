#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike
# Camarilla pivot levels (R3/S3) from prior 1w act as strong support/resistance; breakouts with volume
# indicate institutional participation. 1w EMA50 ensures trend alignment to avoid counter-trend trades.
# Volume confirmation (2.0x 50-period EMA) filters weak breakouts. Designed for 12h timeframe
# to target 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.25).
# Works in bull markets by buying breakouts in uptrends and in bear markets by selling
# breakdowns in downtrends, avoiding range-bound whipsaws.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivot calculation and EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels for each 12h bar using prior 1w bar's OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        # Need prior 1w bar data (1w bar must be closed)
        if i < 14:  # 14*12h = 168h = 1w, need at least one full 1w bar before current 12h bar
            continue
            
        # Get index of prior 1w bar in 1w dataframe
        # 1w bar index = floor(i / 14) - 1 (since we want prior completed 1w bar)
        idx_1w = (i // 14) - 1
        if idx_1w < 0 or idx_1w >= len(df_1w):
            continue
            
        # Calculate Camarilla levels from prior 1w bar
        h_1w = df_1w['high'].iloc[idx_1w]
        l_1w = df_1w['low'].iloc[idx_1w]
        c_1w = df_1w['close'].iloc[idx_1w]
        
        camarilla_r3[i] = c_1w + (h_1w - l_1w) * 1.1 / 4
        camarilla_s3[i] = c_1w - (h_1w - l_1w) * 1.1 / 4
    
    # Volume confirmation: 2.0x 50-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_50 = vol_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start from 200 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 50-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_50[i])
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + price above 1w EMA50 (uptrend)
            if (close[i] > camarilla_r3[i] and volume_spike and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume spike + price below 1w EMA50 (downtrend)
            elif (close[i] < camarilla_s3[i] and volume_spike and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 OR price below 1w EMA50 (trend change)
            if close[i] < camarilla_s3[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 OR price above 1w EMA50 (trend change)
            if close[i] > camarilla_r3[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals