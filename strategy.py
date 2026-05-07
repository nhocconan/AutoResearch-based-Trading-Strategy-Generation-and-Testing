#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
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
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA34 trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    trend_up = close > ema_34_1w_aligned
    trend_down = close < ema_34_1w_aligned
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    camarilla_high = high_prev + (low_prev - close_prev) * 1.1 / 12
    camarilla_low = close_prev - (high_prev - low_prev) * 1.1 / 12
    
    # R3 and S3 levels
    camarilla_r3 = camarilla_high + 4 * (camarilla_high - camarilla_low)
    camarilla_s3 = camarilla_low - 4 * (camarilla_high - camarilla_low)
    
    # Volume spike: current volume > 2.0x 20-period average (~10 days)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~2 days (4*12h) to reduce trade frequency
    
    start_idx = max(1, 20)  # Ensure enough data for Camarilla and volume
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
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
            # Long: Price breaks above Camarilla R3 with volume spike in 1w uptrend
            if (close[i] > camarilla_r3[i] and 
                trending_up and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S3 with volume spike in 1w downtrend
            elif (close[i] < camarilla_s3[i] and 
                  trending_down and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Camarilla S3 or 1w trend changes to down
            if close[i] < camarilla_s3[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above Camarilla R3 or 1w trend changes to up
            if close[i] > camarilla_r3[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 12h timeframe, price breaking above/below Camarilla R3/S3 levels with volume spike confirmation and 1-week EMA34 trend filter captures institutional breakout momentum. Camarilla levels represent key intraday support/resistance derived from previous day's price action, reducing false breakouts. 1w trend filter ensures alignment with higher timeframe momentum, improving performance in both bull and bear markets. Volume spike filter (2.0x 20-period average) confirms institutional participation. Cooldown period prevents overtrading. Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Uses discrete position sizing (0.25) to balance risk and reward while reducing fee churn. This strategy focuses on proven Camarilla breakout with volume/trend confluence, which has shown strong performance in DB. Using 1w trend instead of 1d trend provides stronger trend filter for 12h timeframe, reducing whipsaws and improving generalization.