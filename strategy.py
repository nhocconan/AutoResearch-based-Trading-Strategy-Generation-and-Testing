#!/usr/bin/env python3

name = "4h_KAMA_Trend_Direction_12h_TrendFilter_v2"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # KAMA parameters
    er_len = 10
    fast_len = 2
    slow_len = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.full(n, np.nan)
    for i in range(er_len, n):
        if volatility[i] != 0:
            er[i] = change[i - er_len + 1] / volatility[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_len] = close[er_len]
    for i in range(er_len + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 12h EMA21 for trend filter
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Volume filter: current volume > 1.8x 30-period average (on 4h data)
    vol_ma_30 = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma_30[i] = np.mean(volume[i-30:i])
    vol_filter = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # Prevent overtrading (approx 2 days for 4h)
    
    start_idx = max(30, er_len)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 12h trend direction
        trend_12h_up = close[i] > ema_21_12h_aligned[i]
        trend_12h_down = close[i] < ema_21_12h_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: price above KAMA in 12h uptrend with volume filter
            if (close[i] > kama[i] and 
                trend_12h_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: price below KAMA in 12h downtrend with volume filter
            elif (close[i] < kama[i] and 
                  trend_12h_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: price crosses below KAMA OR trend change
            if (close[i] < kama[i] or not trend_12h_up):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA OR trend change
            if (close[i] > kama[i] or not trend_12h_down):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: This strategy uses Kaufman's Adaptive Moving Average (KAMA) as the primary trend indicator,
# with 12h EMA21 as a higher-timeframe trend filter and volume confirmation to avoid false signals.
# KAMA adapts to market noise, making it effective in both trending and ranging markets.
# The 12h trend filter ensures alignment with medium-term market direction, while volume confirmation
# ensures institutional participation. The cooldown period prevents overtrading. This combination
# should work in both bull and bear markets by adapting to changing volatility conditions.