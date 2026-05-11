#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "6h"
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
    
    # 1w trend: EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    trend_up = close > ema_1w_aligned
    
    # 1d data for weekly pivot (start of week)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points from Monday's data
    # We need to get the first day of the week for each 6h bar
    weekly_high = np.full(n, np.nan)
    weekly_low = np.full(n, np.nan)
    weekly_close = np.full(n, np.nan)
    
    # Convert to pandas for easier resampling (but only for calculation, not signal gen)
    df_temp = pd.DataFrame({
        'high': high,
        'low': low,
        'close': close
    }, index=pd.to_datetime(prices['open_time']))
    
    # Resample to weekly, taking first day's OHLC
    weekly_data = df_temp.resample('W-Mon').agg({'high': 'first', 'low': 'first', 'close': 'first'})
    
    # Forward fill to get weekly values for each 6h bar
    weekly_high_s = weekly_data['high'].reindex(df_temp.index, method='ffill')
    weekly_low_s = weekly_data['low'].reindex(df_temp.index, method='ffill')
    weekly_close_s = weekly_data['close'].reindex(df_temp.index, method='ffill')
    
    weekly_high = weekly_high_s.values
    weekly_low = weekly_low_s.values
    weekly_close = weekly_close_s.values
    
    # Weekly Camarilla R3 and S3
    R3_weekly = weekly_close + (weekly_high - weekly_low) * 1.1 / 4
    S3_weekly = weekly_close - (weekly_high - weekly_low) * 1.1 / 4
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(R3_weekly[i]) or np.isnan(S3_weekly[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > Weekly R3 + 1w uptrend + volume spike
            if close[i] > R3_weekly[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < Weekly S3 + 1w downtrend + volume spike
            elif close[i] < S3_weekly[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < Weekly S3 or 1w trend down
            if close[i] < S3_weekly[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close > Weekly R3 or 1w trend up
            if close[i] > R3_weekly[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals