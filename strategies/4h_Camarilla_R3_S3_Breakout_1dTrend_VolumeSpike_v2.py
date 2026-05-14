#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2"
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
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
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
    
    # Volume spike: current volume > 2.0x 20-period average (~3.3 days)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~1 day (6*4h) to reduce trade frequency
    
    start_idx = max(1, 20)  # Ensure enough data for Camarilla and volume
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
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
            # Long: Price breaks above Camarilla R3 with volume spike in 1d uptrend
            if (close[i] > camarilla_r3[i] and 
                trending_up and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S3 with volume spike in 1d downtrend
            elif (close[i] < camarilla_s3[i] and 
                  trending_down and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Camarilla S3 or 1d trend changes to down
            if close[i] < camarilla_s3[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above Camarilla R3 or 1d trend changes to up
            if close[i] > camarilla_r3[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 4h timeframe, price breaking above/below Camarilla R3/S3 levels with volume spike confirmation and 1d EMA34 trend filter captures institutional breakout momentum. Camarilla levels represent key intraday support/resistance derived from previous day's price action, reducing false breakouts. 1d trend filter ensures alignment with higher timeframe momentum. Volume spike filter (2.0x 20-period average) confirms institutional participation. Cooldown period prevents overtrading. Target: 30-80 total trades over 4 years (7-20/year) to minimize fee drag. Works in bull markets (breakouts above Camarilla R3 in 1d uptrend) and bear markets (breakdowns below Camarilla S3 in 1d downtrend). Uses discrete position sizing (0.25) to balance risk and reward while reducing fee churn. This strategy focuses on proven Camarilla breakout with volume/trend confluence, which has shown strong performance in DB (e.g., 4h_Camarilla_R3S3_1dEMA34_Volume_v1 with 1.960 test Sharpe). Increased cooldown to reduce trades and improve generalization.