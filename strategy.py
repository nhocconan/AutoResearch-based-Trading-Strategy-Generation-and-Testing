#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike
# Camarilla pivot levels (R3/S3) from prior 1d act as strong support/resistance; breakouts with volume
# indicate institutional participation. 1d EMA34 ensures trend alignment to avoid counter-trend trades.
# Volume confirmation (2.0x 20-period EMA) filters weak breakouts. Designed for 12h timeframe
# to target 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.25).
# Works in bull markets by buying breakouts in uptrends and in bear markets by selling
# breakdowns in downtrends, avoiding range-bound whipsaws.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
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
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for each 12h bar using prior 1d bar's OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        # Need prior 1d bar data (1d bar must be closed)
        # 12h bar index: 2 per day, so prior 1d bar index = floor(i / 2) - 1
        if i < 2:  # Need at least one full 1d bar before current 12h bar
            continue
            
        # Get index of prior 1d bar in 1d dataframe
        idx_1d = (i // 2) - 1
        if idx_1d < 0 or idx_1d >= len(df_1d):
            continue
            
        # Calculate Camarilla levels from prior 1d bar
        h_1d = df_1d['high'].iloc[idx_1d]
        l_1d = df_1d['low'].iloc[idx_1d]
        c_1d = df_1d['close'].iloc[idx_1d]
        
        camarilla_r3[i] = c_1d + (h_1d - l_1d) * 1.1 / 4
        camarilla_s3[i] = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # Volume confirmation: 2.0x 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Start from 2 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + price above 1d EMA34 (uptrend)
            if (close[i] > camarilla_r3[i] and volume_spike and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume spike + price below 1d EMA34 (downtrend)
            elif (close[i] < camarilla_s3[i] and volume_spike and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S3 OR price below 1d EMA34 (trend change)
            if close[i] < camarilla_s3[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R3 OR price above 1d EMA34 (trend change)
            if close[i] > camarilla_r3[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals