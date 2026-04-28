#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R1/S1) with volume confirmation and 1w EMA34 trend filter.
# Enter long when price breaks above R1 with volume spike and above 1w EMA34.
# Enter short when price breaks below S1 with volume spike and below 1w EMA34.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-150 total trades over 4 years.
# Works in bull/bear via trend filter (EMA34) and volatility filter (volume spike).

name = "4h_Camarilla_R1S1_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
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
    R1 = np.full(n_1d, np.nan)
    S1 = np.full(n_1d, np.nan)
    PP = np.full(n_1d, np.nan)
    
    for i in range(n_1d):
        # Camarilla pivot calculation
        PP[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        range_1d = high_1d[i] - low_1d[i]
        R1[i] = PP[i] + range_1d * 1.1 / 12.0
        S1[i] = PP[i] - range_1d * 1.1 / 12.0
    
    # Forward fill to get most recent pivot levels
    R1 = pd.Series(R1).ffill().values
    S1 = pd.Series(S1).ffill().values
    PP = pd.Series(PP).ffill().values
    
    # Align 1d Camarilla levels to 4h timeframe with 1-bar delay for confirmation
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to 4h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1w EMA34
        above_ema = close[i] > ema_34_1w_aligned[i]
        below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Camarilla breakout conditions with volume confirmation
        long_breakout = close[i] > R1_aligned[i] and volume_spike[i]
        short_breakout = close[i] < S1_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite pivot level or trend reversal
        long_exit = close[i] < S1_aligned[i] or below_ema
        short_exit = close[i] > R1_aligned[i] or above_ema
        
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