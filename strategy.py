#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels with 1d volume spike and 1d EMA50 trend filter.
# Enter long when price breaks above R3 with volume spike and above 1d EMA50.
# Enter short when price breaks below S3 with volume spike and below 1d EMA50.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-150 total trades over 4 years.
# This combines intraday pivot structure with higher timeframe trend and volume confirmation to work in both bull and bear markets.

name = "6h_Camarilla_R3S3_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Get 12h data for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    n_12h = len(high_12h)
    R3 = np.full(n_12h, np.nan)
    S3 = np.full(n_12h, np.nan)
    PP = np.full(n_12h, np.nan)
    
    for i in range(n_12h):
        # Camarilla pivot calculation
        PP[i] = (high_12h[i] + low_12h[i] + close_12h[i]) / 3.0
        range_12h = high_12h[i] - low_12h[i]
        R3[i] = PP[i] + range_12h * 1.1 / 4.0
        S3[i] = PP[i] - range_12h * 1.1 / 4.0
    
    # Forward fill to get most recent pivot levels
    R3 = pd.Series(R3).ffill().values
    S3 = pd.Series(S3).ffill().values
    PP = pd.Series(PP).ffill().values
    
    # Align 12h Camarilla levels to 6h timeframe with 1-bar delay for confirmation
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    PP_aligned = align_htf_to_ltf(prices, df_12h, PP)
    
    # Get daily data for EMA trend filter and volume spike
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume spike: >2.0x 20-bar average volume
    volume_1d = df_1d['volume'].values
    volume_series_1d = pd.Series(volume_1d)
    volume_ma_20_1d = volume_series_1d.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > 2.0 * volume_ma_20_1d
    
    # Align EMA and volume spike to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA50
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R3_aligned[i] and volume_spike_1d_aligned[i] > 0.5
        short_breakout = close[i] < S3_aligned[i] and volume_spike_1d_aligned[i] > 0.5
        
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