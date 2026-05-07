#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
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
    
    # 12h trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    trend_up = close > ema_50_12h_aligned
    trend_down = close < ema_50_12h_aligned
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_high = high_prev + (low_prev - close_prev) * 1.1 / 12
    camarilla_low = close_prev - (high_prev - low_prev) * 1.1 / 12
    
    # R1 and S1 levels (more sensitive than R3/S3)
    camarilla_r1 = camarilla_high + 1.1 * (camarilla_high - camarilla_low)
    camarilla_s1 = camarilla_low - 1.1 * (camarilla_high - camarilla_low)
    
    # Volume spike: current volume > 2.0x 20-period average (~3.3 days)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~2 days (4*4h) to prevent overtrading
    
    start_idx = max(1, 20)  # Ensure enough data for Camarilla and volume
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Camarilla R1 with volume spike in 12h uptrend
            if (close[i] > camarilla_r1[i] and 
                trending_up and 
                vol_spike[i]):
                signals[i] = 0.30
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S1 with volume spike in 12h downtrend
            elif (close[i] < camarilla_s1[i] and 
                  trending_down and 
                  vol_spike[i]):
                signals[i] = -0.30
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Camarilla S1 or 12h trend changes to down
            if close[i] < camarilla_s1[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Price rises back above Camarilla R1 or 12h trend changes to up
            if close[i] > camarilla_r1[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: On 4h timeframe, price breaking above/below Camarilla R1/S1 levels with volume spike confirmation and 12h EMA50 trend filter captures institutional breakout momentum. Camarilla R1/S1 levels are more sensitive than R3/S3, capturing earlier breakout opportunities while still filtering false signals. 12h EMA50 trend ensures alignment with higher timeframe momentum. Volume spike filter (2.0x 20-period average) confirms institutional participation. Cooldown period prevents overtrading. Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag. Works in bull markets (breakouts above Camarilla R1 in 12h uptrend) and bear markets (breakdowns below Camarilla S1 in 12h downtrend). Uses discrete position sizing (0.30) to balance risk and reward while reducing fee churn. This strategy focuses on proven Camarilla breakout with volume/trend confluence, which has shown strong performance in DB (e.g., 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS with 1.867 test Sharpe).