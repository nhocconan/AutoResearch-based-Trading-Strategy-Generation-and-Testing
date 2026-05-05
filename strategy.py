#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Camarilla pivot levels (S1/R1) breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above R1 AND price > 1d EMA50 AND volume > 2.0 * avg_volume(20) on 4h
# Short when price breaks below S1 AND price < 1d EMA50 AND volume > 2.0 * avg_volume(20) on 4h
# Exit when price reverts to the 1d VWAP (mean reversion) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# Camarilla pivot levels provide high-probability intraday reversal points
# 1d EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "4h_Camarilla_R1S1_Breakout_1dEMA50_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least one completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (High + Low + Close) / 3
    # R1 = Close + ((High - Low) * 1.1 / 12)
    # S1 = Close - ((High - Low) * 1.1 / 12)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    R1 = close_1d + (range_1d * 1.1 / 12.0)
    S1 = close_1d - (range_1d * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe (wait for completed daily bar)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Get 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 AND above 1d EMA50 AND volume confirmation
            if (close[i] > R1_aligned[i] and close[i-1] <= R1_aligned[i-1] and 
                close[i] > ema50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 AND below 1d EMA50 AND volume confirmation
            elif (close[i] < S1_aligned[i] and close[i-1] >= S1_aligned[i-1] and 
                  close[i] < ema50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price reverts to 1d VWAP (using close as proxy) OR volume drops below average
            if close[i] < ema50_1d_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price reverts to 1d VWAP (using close as proxy) OR volume drops below average
            if close[i] > ema50_1d_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals