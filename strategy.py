#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot levels with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above weekly R3 with 1d EMA34 uptrend and volume > 2.0 * avg_volume(20)
# Short when price breaks below weekly S3 with 1d EMA34 downtrend and volume > 2.0 * avg_volume(20)
# Exit when price returns to weekly pivot point (PP) or volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Camarilla provides strong support/resistance levels that work in both bull and bear markets
# 1d EMA34 filters for primary trend alignment to avoid counter-trend trades
# Volume confirmation ensures breakout strength and reduces false signals
# Weekly timeframe avoids noise and captures significant market structure

name = "6h_WeeklyCamarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get weekly data ONCE before loop for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Range = High - Low
    range_1w = high_1w - low_1w
    # Camarilla levels
    r3_1w = pp_1w + range_1w * 1.1 / 4  # R3 = PP + Range * 1.1/4
    s3_1w = pp_1w - range_1w * 1.1 / 4  # S3 = PP - Range * 1.1/4
    r4_1w = pp_1w + range_1w * 1.1 / 2  # R4 = PP + Range * 1.1/2
    s4_1w = pp_1w - range_1w * 1.1 / 2  # S4 = PP - Range * 1.1/2
    
    # Align weekly Camarilla levels to 6h timeframe (wait for completed weekly bar)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
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
        if (np.isnan(pp_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R3, 1d EMA34 uptrend, volume confirmation, in session
            if (close[i] > r3_1w_aligned[i] and close[i-1] <= r3_1w_aligned[i-1] and 
                close[i] > ema34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S3, 1d EMA34 downtrend, volume confirmation, in session
            elif (close[i] < s3_1w_aligned[i] and close[i-1] >= s3_1w_aligned[i-1] and 
                  close[i] < ema34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly pivot point (PP) OR volume drops below average
            if close[i] <= pp_1w_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly pivot point (PP) OR volume drops below average
            if close[i] >= pp_1w_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals