#!/usr/bin/env python3
name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    # Use previous day's data to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # First bar uses current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * range_val / 2
    camarilla_s3 = prev_close - 1.1 * range_val / 2
    
    # Volume filter: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Start from second bar (need previous day data)
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above R3 with weekly uptrend and volume
            if (close[i] > camarilla_r3[i] and close[i] > ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close below S3 with weekly downtrend and volume
            elif (close[i] < camarilla_s3[i] and close[i] < ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below R3 or trend change to down
            if close[i] < camarilla_r3[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above S3 or trend change to up
            if close[i] > camarilla_s3[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Camarilla R3/S3 breakout with weekly EMA(50) trend filter and volume confirmation.
# Camarilla levels identify key support/resistance based on previous day's range.
# In weekly uptrends, buy breakouts above R3; in weekly downtrends, sell breakdowns below S3.
# Volume filter ensures institutional participation. Position size 0.25 controls risk.
# Weekly trend filter adapts to multi-week market regime, working in both bull and bear markets.