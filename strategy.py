#!/usr/bin/env python3
name = "1d_R3S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA(10) for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily close for Camarilla calculation (previous day)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    prev_high = np.roll(high, 1)
    prev_high[0] = np.nan
    prev_low = np.roll(low, 1)
    prev_low[0] = np.nan
    
    # R3 and S3 levels: Close +/- 1.1 * (High - Low)
    r3 = prev_close + 1.1 * (prev_high - prev_low)
    s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Volume filter: > 1.5x 10-period average
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Wait for weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with weekly uptrend and volume
            if (close[i] > r3[i] and close[i] > ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with weekly downtrend and volume
            elif (close[i] < s3[i] and close[i] < ema_1w_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below S3 or weekly trend change
            if close[i] < s3[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above R3 or weekly trend change
            if close[i] > r3[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily Camarilla R3/S3 breakout with weekly EMA(10) trend filter and volume confirmation.
# Camarilla levels identify key support/resistance from prior day. Breaking R3/S3 indicates strong momentum.
# Weekly EMA filter ensures alignment with higher timeframe trend. Volume confirms institutional participation.
# Target: 10-25 trades/year to minimize fee drift. Position size 0.25 limits drawdown in volatile markets.