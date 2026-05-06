#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly and daily pivot points with volume confirmation
# Weekly pivots (R1/S1) define major weekly support/resistance, daily pivots provide intraday levels
# Breakout above weekly R1 or below weekly S1 with volume > 2x 20-period average indicates strong momentum
# Rejection at weekly R2/S2 with volume confirmation indicates mean reversion within weekly range
# Uses 60-period EMA on 6h timeframe for trend filter to avoid counter-trend trades
# Works in bull/bear markets: breakouts capture trends, reversals capture pullbacks within trend
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WeeklyDailyPivot_R1S2_VolumeTrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly pivot points ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous week's OHLC for pivot calculation
    prev_close_w = df_1w['close'].shift(1).values
    prev_high_w = df_1w['high'].shift(1).values
    prev_low_w = df_1w['low'].shift(1).values
    
    # Weekly pivot point calculation
    pivot_w = (prev_high_w + prev_low_w + prev_close_w) / 3
    range_w = prev_high_w - prev_low_w
    
    # Weekly Support and Resistance levels
    r1_w = pivot_w + (range_w * 1.0)
    r2_w = pivot_w + (range_w * 2.0)
    s1_w = pivot_w - (range_w * 1.0)
    s2_w = pivot_w - (range_w * 2.0)
    
    # Align weekly levels to 6h timeframe
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # Calculate daily pivot points ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for pivot calculation
    prev_close_d = df_1d['close'].shift(1).values
    prev_high_d = df_1d['high'].shift(1).values
    prev_low_d = df_1d['low'].shift(1).values
    
    # Daily pivot point calculation
    pivot_d = (prev_high_d + prev_low_d + prev_close_d) / 3
    range_d = prev_high_d - prev_low_d
    
    # Daily Support and Resistance levels
    r1_d = pivot_d + (range_d * 1.0)
    r2_d = pivot_d + (range_d * 2.0)
    s1_d = pivot_d - (range_d * 1.0)
    s2_d = pivot_d - (range_d * 2.0)
    
    # Align daily levels to 6h timeframe
    r1_d_aligned = align_htf_to_ltf(prices, df_1d, r1_d)
    r2_d_aligned = align_htf_to_ltf(prices, df_1d, r2_d)
    s1_d_aligned = align_htf_to_ltf(prices, df_1d, s1_d)
    s2_d_aligned = align_htf_to_ltf(prices, df_1d, s2_d)
    
    # Volume confirmation: >2.0x 20-period average (higher threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Trend filter: 60-period EMA on 6h timeframe
    close_series = pd.Series(close)
    ema_60 = close_series.ewm(span=60, adjust=False, min_periods=60).mean().values
    uptrend = close > ema_60
    downtrend = close < ema_60
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(r1_w_aligned[i]) or np.isnan(r2_w_aligned[i]) or np.isnan(s1_w_aligned[i]) or 
            np.isnan(s2_w_aligned[i]) or np.isnan(r1_d_aligned[i]) or np.isnan(r2_d_aligned[i]) or 
            np.isnan(s1_d_aligned[i]) or np.isnan(s2_d_aligned[i]) or np.isnan(volume_filter[i]) or 
            np.isnan(ema_60[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly R1 with volume confirmation and uptrend
            if close[i] > r1_w_aligned[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below weekly S1 with volume confirmation and downtrend
            elif close[i] < s1_w_aligned[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects weekly S2 with volume confirmation (bounce from support)
            elif close[i] < s2_w_aligned[i] and close[i] > s2_w_aligned[i] * 0.995 and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects weekly R2 with volume confirmation (rejection from resistance)
            elif close[i] > r2_w_aligned[i] and close[i] < r2_w_aligned[i] * 1.005 and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly S1 (failed support) or reaches weekly R2 (take profit)
            if close[i] < s1_w_aligned[i] or close[i] > r2_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly R1 (failed resistance) or reaches weekly S2 (take profit)
            if close[i] > r1_w_aligned[i] or close[i] < s2_w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
#%%