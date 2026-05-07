#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSurge_v2"
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
    
    # Daily trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Daily EMA34 trend
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    # Daily OHLC for Camarilla R3/S3 levels
    camarilla_r3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_s3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume surge filter: current volume > 2.5x 8-period average (4-day equivalent in 12h)
    vol_ma_8 = np.full(n, np.nan)
    for i in range(8, n):
        vol_ma_8[i] = np.mean(volume[i-8:i])
    vol_surge = volume > (2.5 * vol_ma_8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~1.5 days (3*12h) to prevent overtrading
    
    start_idx = max(8, 34)  # Ensure enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
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
            # Long: Price breaks above Camarilla R3 with volume surge in daily uptrend
            if (close[i] > r3_aligned[i] and 
                trending_up and 
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S3 with volume surge in daily downtrend
            elif (close[i] < s3_aligned[i] and 
                  trending_down and 
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Camarilla S3 or daily trend changes to down
            if close[i] < s3_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above Camarilla R3 or daily trend changes to up
            if close[i] > r3_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 12h timeframe, price breaking above/below Camarilla R3/S3 levels with volume surge confirmation and daily EMA34 trend filter captures institutional breakout momentum. Camarilla R3/S3 represent stronger support/resistance, reducing false breakouts. Daily trend filter ensures alignment with higher timeframe momentum. Volume surge filter (2.5x 8-period average) confirms institutional participation. Increased cooldown (3 bars) and higher volume threshold (2.5x) reduce trade frequency to target 50-150 total trades over 4 years. Position size reduced to 0.25 to manage drawdown. Works in bull markets (breakouts above R3 in daily uptrend) and bear markets (breakdowns below S3 in daily downtrend). Uses discrete position sizing (0.25) to balance risk and reward while minimizing fee churn.