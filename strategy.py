#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R1 AND price > 1d EMA50 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below Camarilla S1 AND price < 1d EMA50 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses Camarilla pivot point (mean reversion) OR volume drops below average
# Uses discrete sizing 0.20 to balance return and risk
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Camarilla pivots provide intraday support/resistance levels that work in ranging markets
# 1d EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "1h_Camarilla_R1S1_Breakout_1dEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # Need at least one completed 4h bar for pivots
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivots for 4h timeframe
    # Pivot = (High + Low + Close) / 3
    # R1 = Close + 1.1 * (High - Low) / 12
    # S1 = Close - 1.1 * (High - Low) / 12
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    r1_4h = close_4h + (1.1 * (high_4h - low_4h) / 12.0)
    s1_4h = close_4h - (1.1 * (high_4h - low_4h) / 12.0)
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1, above 1d EMA50, volume confirmation, in session
            if (close[i] > r1_4h_aligned[i] and close[i-1] <= r1_4h_aligned[i-1] and 
                close[i] > ema50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S1, below 1d EMA50, volume confirmation, in session
            elif (close[i] < s1_4h_aligned[i] and close[i-1] >= s1_4h_aligned[i-1] and 
                  close[i] < ema50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla pivot (mean reversion) OR volume drops below average
            if close[i] < pivot_4h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above Camarilla pivot (mean reversion) OR volume drops below average
            if close[i] > pivot_4h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals