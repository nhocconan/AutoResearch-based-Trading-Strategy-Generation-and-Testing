#!/usr/bin/env python3
"""
4h_12h_TRIX_Volume_Regime_v1
Hypothesis: On 4h timeframe, use TRIX momentum with 12h trend filter and volume confirmation.
Long when TRIX crosses above zero with 12h bullish trend and volume spike.
Short when TRIX crosses below zero with 12h bearish trend and volume spike.
Exit when TRIX crosses back in opposite direction.
Designed for low trade frequency (20-40/year) by requiring multiple confluence factors.
Works in bull/bear via 12h trend filter and momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_TRIX_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h TRIX INDICATOR ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # TRIX: triple exponential moving average
    ema1 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    # Calculate TRIX: (ema3 - ema3_prev) / ema3_prev * 100
    ema3_prev = np.roll(ema3, 1)
    ema3_prev[0] = 0
    trix = np.where(ema3_prev != 0, (ema3 - ema3_prev) / ema3_prev * 100, 0)
    
    # === 12h TREND FILTER (EMA CROSSOVER) ===
    ema_fast = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean()
    ema_slow = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean()
    trend_bullish = ema_fast > ema_slow
    trend_bearish = ema_fast < ema_slow
    
    # === 4h VOLUME CONFIRMATION ===
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    # Align 12h indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, trend_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(trix_aligned[i]) or np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # TRIX crossover signals
        trix_cross_up = trix_aligned[i] > 0 and (i == 50 or trix_aligned[i-1] <= 0)
        trix_cross_down = trix_aligned[i] < 0 and (i == 50 or trix_aligned[i-1] >= 0)
        
        # Entry conditions
        long_setup = trix_cross_up and trend_bullish_aligned[i] > 0.5 and vol_confirm
        short_setup = trix_cross_down and trend_bearish_aligned[i] > 0.5 and vol_confirm
        
        # Exit conditions: TRIX crosses back in opposite direction
        exit_long = trix_aligned[i] < 0 and (i == 50 or trix_aligned[i-1] >= 0)
        exit_short = trix_aligned[i] > 0 and (i == 50 or trix_aligned[i-1] <= 0)
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals