#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot R3/S3 breakout with volume confirmation and 1d EMA34 trend filter.
# Enter long when price breaks above 1d Camarilla R3 level with volume spike and above 1d EMA34.
# Enter short when price breaks below 1d Camarilla S3 level with volume spike and below 1d EMA34.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
# Camarilla pivots provide structure, volume confirms breakout strength, EMA34 filters trend direction.
# Works in bull (breakouts with trend) and bear (failed breaks reverse) markets.

name = "12h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    camarilla_r3 = np.full(n_1d, np.nan)
    camarilla_s3 = np.full(n_1d, np.nan)
    pivot = np.full(n_1d, np.nan)
    rang = np.full(n_1d, np.nan)
    
    for i in range(n_1d):
        pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        rang[i] = high_1d[i] - low_1d[i]
        camarilla_r3[i] = pivot[i] + 1.1 * rang[i] / 2.0  # R3 = pivot + 1.1*(high-low)/2
        camarilla_s3[i] = pivot[i] - 1.1 * rang[i] / 2.0  # S3 = pivot - 1.1*(high-low)/2
    
    # Forward fill Camarilla levels
    camarilla_r3 = pd.Series(camarilla_r3).ffill().values
    camarilla_s3 = pd.Series(camarilla_s3).ffill().values
    pivot = pd.Series(pivot).ffill().values
    rang = pd.Series(rang).ffill().values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h volume spike: >2.0x 24-bar average volume (12h equivalent, ~12 bars)
    volume_series = pd.Series(volume)
    volume_ma_12 = volume_series.rolling(window=12, min_periods=12).mean().values
    volume_spike = volume > 2.0 * volume_ma_12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_12[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA34
        above_ema = close[i] > ema_34_1d_aligned[i]
        below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > camarilla_r3_aligned[i] and volume_spike[i]
        short_breakout = close[i] < camarilla_s3_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Camarilla level or trend reversal
        long_exit = close[i] < camarilla_s3_aligned[i] or below_ema
        short_exit = close[i] > camarilla_r3_aligned[i] or above_ema
        
        # Handle entries and exits
        if long_breakout and above_ema and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and below_ema and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals