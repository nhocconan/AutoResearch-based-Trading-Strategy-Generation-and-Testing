#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion
# 1d EMA50 filter ensures we only trade in the direction of higher timeframe trend
# Volume confirmation avoids low-conviction reversals
# Works in bull/bear: mean reversion in ranges, trend-filtered in strong moves
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_williamsr_volume_trend_v1"
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
    
    # Load 1d data ONCE before loop for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d average volume (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar close)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close) / (highest_high - lowest_low),
                          -50)  # neutral when range is zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.3x 1d average volume
        volume_confirmed = volume[i] > 1.3 * avg_volume_1d_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R exits oversold territory OR trend breaks down
            if williams_r[i] > -80 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R exits overbought territory OR trend breaks up
            if williams_r[i] < -20 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R extremes with trend and volume confirmation
            if williams_r[i] <= -90 and uptrend and volume_confirmed:
                # Deep oversold in uptrend -> long
                position = 1
                signals[i] = 0.25
            elif williams_r[i] >= -10 and downtrend and volume_confirmed:
                # Deep overbought in downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals