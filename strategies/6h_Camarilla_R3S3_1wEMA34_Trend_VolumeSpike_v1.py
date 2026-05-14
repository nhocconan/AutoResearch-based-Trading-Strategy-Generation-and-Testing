#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation and 1w EMA34 trend filter.
# Enter long when price breaks above R3 with volume spike and above 1w EMA34.
# Enter short when price breaks below S3 with volume spike and below 1w EMA34.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-150 total trades over 4 years.

name = "6h_Camarilla_R3S3_1wEMA34_Trend_VolumeSpike_v1"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    R3 = np.full(n_1d, np.nan)
    S3 = np.full(n_1d, np.nan)
    R4 = np.full(n_1d, np.nan)
    S4 = np.full(n_1d, np.nan)
    PP = np.full(n_1d, np.nan)
    
    for i in range(n_1d):
        # Camarilla pivot calculation
        PP[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        range_1d = high_1d[i] - low_1d[i]
        R3[i] = PP[i] + range_1d * 1.1 / 4.0
        S3[i] = PP[i] - range_1d * 1.1 / 4.0
        R4[i] = PP[i] + range_1d * 1.1 / 2.0
        S4[i] = PP[i] - range_1d * 1.1 / 2.0
    
    # Forward fill to get most recent pivot levels
    R3 = pd.Series(R3).ffill().values
    S3 = pd.Series(S3).ffill().values
    R4 = pd.Series(R4).ffill().values
    S4 = pd.Series(S4).ffill().values
    PP = pd.Series(PP).ffill().values
    
    # Align 1d Camarilla levels to 6h timeframe with 1-bar delay for confirmation
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
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
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(PP_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1w EMA34
        above_ema = close[i] > ema_34_1w_aligned[i]
        below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
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