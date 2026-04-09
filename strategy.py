#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions for mean reversion
# 12h EMA50 provides trend filter: only take reversals in direction of higher timeframe trend
# Volume spike confirms reversal authenticity (avoids false signals)
# Works in bull/bear: trend filter adapts, Williams %R captures exhaustion moves
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_12h_williamsr_volume_trend_v1"
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
    
    # Load 12h data ONCE before loop for EMA trend and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h average volume (20-period) for confirmation
    volume_12h = df_12h['volume'].values
    avg_volume_12h = pd.Series(volume_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close) / (highest_high - lowest_low),
                          -50)  # neutral when range is zero
    
    # Align 12h indicators to 6h timeframe (wait for 12h bar close)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume_12h_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.8x 12h average volume
        volume_confirmed = volume[i] > 1.8 * avg_volume_12h_aligned[i]
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Williams %R exits oversold territory OR trend turns down
            if williams_r[i] > -80 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R exits overbought territory OR trend turns up
            if williams_r[i] < -20 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Williams %R reversal with trend and volume confirmation
            if williams_r[i] <= -90 and uptrend and volume_confirmed:
                # Deep oversold in uptrend -> long reversal
                position = 1
                signals[i] = 0.25
            elif williams_r[i] >= -10 and downtrend and volume_confirmed:
                # Deep overbought in downtrend -> short reversal
                position = -1
                signals[i] = -0.25
    
    return signals