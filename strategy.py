#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels (R3/S3) with volume confirmation and 1d EMA34 trend filter.
# Enter long when price breaks above weekly R3 with volume spike and above 1d EMA34.
# Enter short when price breaks below weekly S3 with volume spike and below 1d EMA34.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
# Weekly pivot structure provides robust support/resistance that works in both bull and bear markets.
# Volume confirmation reduces false breakouts. EMA34 filter ensures trades align with higher timeframe trend.

name = "6h_Camarilla_W_R3S3_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    n_1w = len(high_1w)
    R3 = np.full(n_1w, np.nan)
    S3 = np.full(n_1w, np.nan)
    PP = np.full(n_1w, np.nan)
    
    for i in range(n_1w):
        # Camarilla pivot calculation (focus on R3/S3 levels)
        PP[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        range_1w = high_1w[i] - low_1w[i]
        R3[i] = PP[i] + range_1w * 1.1 / 4.0
        S3[i] = PP[i] - range_1w * 1.1 / 4.0
    
    # Forward fill to get most recent pivot levels
    R3 = pd.Series(R3).ffill().values
    S3 = pd.Series(S3).ffill().values
    PP = pd.Series(PP).ffill().values
    
    # Align weekly Camarilla levels to 6h timeframe with 1-bar delay for confirmation
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    PP_aligned = align_htf_to_ltf(prices, df_1w, PP)
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA34
        above_ema = close[i] > ema_34_1d_aligned[i]
        below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Weekly Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R3_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S3_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite pivot level or trend reversal
        long_exit = close[i] < S3_aligned[i] or below_ema
        short_exit = close[i] > R3_aligned[i] or above_ema
        
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