#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_1w_volume_v1
Hypothesis: On 12-hour timeframe, use Camarilla pivot levels from 1-day timeframe with 1-week trend filter and volume confirmation.
Long when price touches S3 level with weekly EMA(50) trending up and volume > 2x 20-period average.
Short when price touches R3 level with weekly EMA(50) trending down and volume > 2x 20-period average.
Exit when price reaches S4/R4 or opposite pivot level.
Designed for 15-25 trades/year to minimize fee decay while capturing mean reversion at extreme levels.
Works in both bull/bear markets as Camarilla adapts to volatility and weekly trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_1w_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's OHLC for Camarilla
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    camarilla_s3 = prev_close - (1.1 * range_ / 6)
    camarilla_s4 = prev_close - (1.1 * range_ / 4)
    camarilla_r3 = prev_close + (1.1 * range_ / 6)
    camarilla_r4 = prev_close + (1.1 * range_ / 4)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Determine weekly trend direction (using EMA slope)
    weekly_trend_up = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    weekly_trend_down = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    for i in range(1, len(ema_50_1w_aligned)):
        if not np.isnan(ema_50_1w_aligned[i]) and not np.isnan(ema_50_1w_aligned[i-1]):
            weekly_trend_up[i] = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_trend_down[i] = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not available
        if (np.isnan(camarilla_s3[i-1]) or np.isnan(camarilla_s4[i-1]) or 
            np.isnan(camarilla_r3[i-1]) or np.isnan(camarilla_r4[i-1]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S4 or R4
            if close[i] <= camarilla_s4[i-1] or close[i] >= camarilla_r4[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R4 or S4
            if close[i] >= camarilla_r4[i-1] or close[i] <= camarilla_s4[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and weekly trend alignment
            if vol_ok:
                # Long: price touches S3 with weekly uptrend
                if (close[i] <= camarilla_s3[i-1] and close[i-1] > camarilla_s3[i-1] and 
                    weekly_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price touches R3 with weekly downtrend
                elif (close[i] >= camarilla_r3[i-1] and close[i-1] < camarilla_r3[i-1] and 
                      weekly_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals