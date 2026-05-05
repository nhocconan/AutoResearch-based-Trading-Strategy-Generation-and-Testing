#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Camarilla pivot breakouts with 12h EMA50 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 level AND price > 12h EMA50 AND volume > 2.0 * avg_volume(20) on 4h
# Short when price breaks below Camarilla S3 level AND price < 12h EMA50 AND volume > 2.0 * avg_volume(20) on 4h
# Exit when price retests Camarilla pivot level (PP) OR volume drops below average
# Uses discrete sizing 0.30 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels provide precise intraday support/resistance based on prior day's range
# 12h EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike"
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
    if len(df_1d) < 1:  # Need at least one completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from prior 1d bar
    # PP = (High + Low + Close) / 3
    # R3 = PP + (High - Low) * 1.1 / 2
    # S3 = PP - (High - Low) * 1.1 / 2
    pp = (high_1d + low_1d + close_1d) / 3.0
    r3 = pp + (high_1d - low_1d) * 1.1 / 2.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (wait for completed daily bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
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
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, above 12h EMA50, volume confirmation, in session
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                close[i] > ema50_12h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3, below 12h EMA50, volume confirmation, in session
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                  close[i] < ema50_12h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price retests pivot point (PP) OR volume drops below average
            if close[i] <= pp_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price retests pivot point (PP) OR volume drops below average
            if close[i] >= pp_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals