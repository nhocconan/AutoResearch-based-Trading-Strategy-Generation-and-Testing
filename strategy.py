#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume
# Hypothesis: Combines Camarilla pivot breakout at R3/S3 with 1-day EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 with volume > 1.5x average and price > 1-day EMA34.
# Short when price breaks below S3 with volume > 1.5x average and price < 1-day EMA34.
# Exits when price crosses EMA34 in opposite direction.
# Designed for 30-50 trades/year to avoid overtrading and work in both bull and bear markets.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
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
    
    # Camarilla Pivot Levels (based on previous day)
    # Calculate once per day using previous day's OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    # Get unique dates from open_time
    dates = pd.to_datetime(prices['open_time']).date
    unique_dates = np.unique(dates)
    
    # Pre-calculate daily OHLC
    daily_open = np.full(n, np.nan)
    daily_high = np.full(n, np.nan)
    daily_low = np.full(n, np.nan)
    daily_close = np.full(n, np.nan)
    
    for date in unique_dates:
        mask = (dates == date)
        if np.any(mask):
            idx = np.where(mask)[0]
            daily_open[idx] = open_prices[mask][0]
            daily_high[idx] = np.max(high[mask])
            daily_low[idx] = np.min(low[mask])
            daily_close[idx] = close_prices[mask][-1]
    
    # Calculate Camarilla levels for each bar using previous day's data
    for i in range(1, n):
        if not np.isnan(daily_close[i-1]) and not np.isnan(daily_high[i-1]) and not np.isnan(daily_low[i-1]):
            prev_close = daily_close[i-1]
            prev_high = daily_high[i-1]
            prev_low = daily_low[i-1]
            camarilla_r3[i] = prev_close + 1.1 * (prev_high - prev_low) / 12
            camarilla_s3[i] = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Volume average (20)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # 1-day EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R3 with volume confirmation and uptrend
            if close[i] > camarilla_r3[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S3 with volume confirmation and downtrend
            elif close[i] < camarilla_s3[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price crosses below EMA34
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price crosses above EMA34
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals