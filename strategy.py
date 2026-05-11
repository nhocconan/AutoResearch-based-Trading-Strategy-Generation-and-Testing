#!/usr/bin/env python3
# 1h_4d_Camarilla_R3_S3_Breakout_Trend_Volume
# Hypothesis: Uses daily and 4-day (96h) high/low for breakout levels, with trend filter from 4h EMA34.
# Breakouts above 4-day high or below 4-day low are taken with volume confirmation and trend alignment.
# In bull markets: breakouts above 4-day high in uptrend capture momentum.
# In bear markets: breakdowns below 4-day low in downtrend capture short moves.
# Volume filter ensures breakouts have conviction, reducing false signals.
# Uses 4h/1d for signal direction, 1h only for entry timing to reduce noise.
# Target: 15-37 trades/year to minimize fee drag while capturing meaningful moves.

name = "1h_4d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h and 1d data for multi-timeframe analysis
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 34 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 4-day (96h) high/low from 4h data ---
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4-period (4*4h=16h) rolling high/low for 4-day equivalent
    # Using 6 periods of 4h = 24h, so 4 days = 6*4 = 24 periods
    period_4d = 6  # 6 * 4h = 24h, but we want 4 days = 96h = 24 * 4h
    high_4d = pd.Series(high_4h).rolling(window=period_4d*4, min_periods=period_4d*4).max().values  # 24 periods for 4 days
    low_4d = pd.Series(low_4h).rolling(window=period_4d*4, min_periods=period_4d*4).min().values
    
    # Shift by 1 to use only completed 4-day period (avoid look-ahead)
    high_4d_prev = np.roll(high_4d, 1)
    low_4d_prev = np.roll(low_4d, 1)
    high_4d_prev[0] = np.nan
    low_4d_prev[0] = np.nan
    
    # Align 4-day high/low to 1h
    high_4d_aligned = align_htf_to_ltf(prices, df_4h, high_4d_prev)
    low_4d_aligned = align_htf_to_ltf(prices, df_4h, low_4d_prev)
    
    # --- 4h EMA34 for trend filter ---
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # --- Volume confirmation (1.5x 20-period average on 1h) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 4-day calculation (96 periods of 1h) and 4h EMA34
    start_idx = 96
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any critical values are NaN
        if (np.isnan(high_4d_aligned[i]) or
            np.isnan(low_4d_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above 4-day high with volume surge and 4h uptrend
            if (close[i] > high_4d_aligned[i] and 
                volume_surge and 
                ema_34_aligned[i] < close[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4-day low with volume surge and 4h downtrend
            elif (close[i] < low_4d_aligned[i] and 
                  volume_surge and 
                  ema_34_aligned[i] > close[i]):
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Exit long: price drops below 4-day low OR 4h EMA34 turns down
                if (close[i] < low_4d_aligned[i] or 
                    close[i] < ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price rises above 4-day high OR 4h EMA34 turns up
                if (close[i] > high_4d_aligned[i] or 
                    close[i] > ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals