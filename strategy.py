#!/usr/bin/env python3
# 1d_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume
# Hypothesis: On 1d chart, enter long when price breaks above Camarilla R3 with volume confirmation and weekly EMA34 trend up,
# enter short when price breaks below S3 with volume confirmation and weekly EMA34 trend down.
# Uses weekly trend filter to avoid counter-trend trades, volume confirmation to reduce false breakouts.
# Designed for low trade frequency (~10-25/year) to minimize fee drag and work in trending and ranging markets.
# Camarilla levels provide precise support/resistance; weekly trend ensures alignment with higher timeframe momentum.
# Works in both bull and bear markets by capturing breakouts in the direction of the weekly trend.
timeframe = "1d"
name = "1d_Camarilla_R3_S3_Breakout_WeeklyTrend_Volume"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla parameters (based on previous day)
    lookback = 1
    
    # Calculate previous day's high, low, close
    prev_high = np.roll(high, lookback)
    prev_low = np.roll(low, lookback)
    prev_close = np.roll(close, lookback)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels for today based on yesterday's range
    range_val = prev_high - prev_low
    camarilla_r3 = prev_close + range_val * 1.1 / 4
    camarilla_s3 = prev_close - range_val * 1.1 / 4
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup for volume MA
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or np.isnan(weekly_ema_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + volume spike + weekly uptrend
            if (close[i] > camarilla_r3[i] and 
                volume[i] > 2.0 * vol_ma[i] and 
                close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + volume spike + weekly downtrend
            elif (close[i] < camarilla_s3[i] and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  close[i] < weekly_ema_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below Camarilla S3 (mean reversion) or weekly trend turns down
            if close[i] < camarilla_s3[i] or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above Camarilla R3 (mean reversion) or weekly trend turns up
            if close[i] > camarilla_r3[i] or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals