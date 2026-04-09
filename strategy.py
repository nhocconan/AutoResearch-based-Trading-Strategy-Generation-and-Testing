#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) extreme + 1d volume spike + 1w trend filter
# Williams %R identifies overbought/oversold conditions for mean reversion
# Volume spike confirms participation at extremes (avoids fake reversals)
# 1w trend filter ensures we only trade counter-trend in strong trends (fade strength)
# Works in bull/bear: mean reversion at extremes works in both regimes when volume confirms
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_1d_1w_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    highest_high_1d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) != 0,
                             -100 * (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d),
                             -50)  # neutral when range is zero
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    volume_s_1d = pd.Series(volume_1d)
    avg_volume_1d = volume_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar close)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Align 1w EMA to 6h timeframe (wait for 1w bar close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 2.0x 1d average volume
        volume_confirmed = volume[i] > 2.0 * avg_volume_1d_aligned[i]
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R returns above -20 (overbought) OR volume confirmation fails
            if williams_r_1d_aligned[i] > -20 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R returns below -80 (oversold) OR volume confirmation fails
            if williams_r_1d_aligned[i] < -80 or not volume_confirmed:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: extreme Williams %R with volume confirmation
            # Only trade counter to weekly trend (fade strength)
            if volume_confirmed:
                if williams_r_1d_aligned[i] < -80 and downtrend:  # Oversold in downtrend = long
                    position = 1
                    signals[i] = 0.25
                elif williams_r_1d_aligned[i] > -20 and uptrend:   # Overbought in uptrend = short
                    position = -1
                    signals[i] = -0.25
    
    return signals