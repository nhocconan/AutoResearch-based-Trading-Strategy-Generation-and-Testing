#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla pivot breakout with daily trend filter and volume confirmation
# Long when price breaks above weekly R4 AND price > daily EMA50 AND volume > 2.0 * avg_volume(20) on 6h
# Short when price breaks below weekly S4 AND price < daily EMA50 AND volume > 2.0 * avg_volume(20) on 6h
# Exit when price returns to weekly midpoint (PP) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Camarilla provides strong support/resistance levels that work in both bull and bear markets
# Daily EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "6h_Camarilla_Weekly_R4S4_Breakout_DailyEMA50_VolumeSpike"
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
    if len(df_1w) < 2:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    # Range = High - Low
    range_1w = high_1w - low_1w
    # Resistance levels: R1 = PP + (Range * 1.1/12), R2 = PP + (Range * 1.1/6), R3 = PP + (Range * 1.1/4), R4 = PP + (Range * 1.1/2)
    # Support levels: S1 = PP - (Range * 1.1/12), S2 = PP - (Range * 1.1/6), S3 = PP - (Range * 1.1/4), S4 = PP - (Range * 1.1/2)
    r1_1w = pp_1w + (range_1w * 1.1 / 12)
    r2_1w = pp_1w + (range_1w * 1.1 / 6)
    r3_1w = pp_1w + (range_1w * 1.1 / 4)
    r4_1w = pp_1w + (range_1w * 1.1 / 2)
    s1_1w = pp_1w - (range_1w * 1.1 / 12)
    s2_1w = pp_1w - (range_1w * 1.1 / 6)
    s3_1w = pp_1w - (range_1w * 1.1 / 4)
    s4_1w = pp_1w - (range_1w * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe (wait for completed weekly bar)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Get daily data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or np.isnan(pp_1w_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly R4 AND above daily EMA50 AND volume confirmation AND in session
            if (close[i] > r4_1w_aligned[i] and close[i-1] <= r4_1w_aligned[i-1] and 
                close[i] > ema50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4 AND below daily EMA50 AND volume confirmation AND in session
            elif (close[i] < s4_1w_aligned[i] and close[i-1] >= s4_1w_aligned[i-1] and 
                  close[i] < ema50_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly midpoint (PP) OR volume drops below average
            if close[i] <= pp_1w_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly midpoint (PP) OR volume drops below average
            if close[i] >= pp_1w_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals