#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camarilla_R3S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for 12h price context and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation (R3, S3)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    prev_close[0] = df_1d['close'].values[0]
    
    # Camarilla pivot levels calculation (R3, S3)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_val = prev_high - prev_low
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Previous week's OHLC for weekly trend (Monday open, Friday close)
    prev_week_open = np.roll(df_1w['open'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_open[0] = df_1w['open'].values[0]
    prev_week_close[0] = df_1w['close'].values[0]
    weekly_trend = prev_week_close > prev_week_open  # True for uptrend, False for downtrend
    
    # Align weekly trend to 12h timeframe
    weekly_trend_12h = align_htf_to_ltf(prices, df_1w, weekly_trend.astype(float))
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(weekly_trend_12h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above R3 with volume spike and weekly uptrend
            long_cond = (close[i] > r3_12h[i] and vol_spike[i] and weekly_trend_12h[i] > 0.5)
            
            # Short entry: price breaks below S3 with volume spike and weekly downtrend
            short_cond = (close[i] < s3_12h[i] and vol_spike[i] and weekly_trend_12h[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (reversal signal)
            if close[i] < s3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses back above R3 (reversal signal)
            if close[i] > r3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout strategy on 12h timeframe with volume spike confirmation and weekly trend filter.
# Enters long when price breaks above R3 with volume spike and weekly uptrend (weekly close > weekly open).
# Enters short when price breaks below S3 with volume spike and weekly downtrend (weekly close < weekly open).
# Exits when price reverses back through S3/R3 respectively.
# Uses R3/S3 levels (wider bands) for fewer, higher-quality breaks vs R1/S1.
# Weekly trend filter ensures alignment with higher timeframe trend, reducing whipsaw.
# Volume spike confirms institutional participation in breakouts.
# Designed for 12-30 trades/year to minimize fee drag while capturing major moves.
# Works in bull markets (trend-following breaks) and bear markets (mean-reversion from extremes).