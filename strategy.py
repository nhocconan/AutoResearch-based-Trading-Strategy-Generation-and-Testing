#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
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
    
    # 4h trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    daily_close_4h = df_4h['close'].values
    daily_high_4h = df_4h['high'].values
    daily_low_4h = df_4h['low'].values
    
    # 4h EMA34 trend
    ema_34_4h = pd.Series(daily_close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    trend_up = close > ema_34_4h_aligned
    trend_down = close < ema_34_4h_aligned
    
    # 4h OHLC for Camarilla R3/S3 levels
    camarilla_r3_4h = daily_close_4h + (daily_high_4h - daily_low_4h) * 1.1 / 4
    camarilla_s3_4h = daily_close_4h - (daily_high_4h - daily_low_4h) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Volume surge filter: current volume > 2.0x 12-period average (6-hour equivalent in 1h)
    vol_ma_12 = np.full(n, np.nan)
    for i in range(12, n):
        vol_ma_12[i] = np.mean(volume[i-12:i])
    vol_surge = volume > (2.0 * vol_ma_12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~4 hours to prevent overtrading
    
    start_idx = max(12, 34)  # Ensure enough data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_4h_aligned[i]) or 
            np.isnan(s3_4h_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i])):
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
            # Long: Price breaks above Camarilla R3 with volume surge in 4h uptrend
            if (close[i] > r3_4h_aligned[i] and 
                trending_up and 
                vol_surge[i]):
                signals[i] = 0.20
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S3 with volume surge in 4h downtrend
            elif (close[i] < s3_4h_aligned[i] and 
                  trending_down and 
                  vol_surge[i]):
                signals[i] = -0.20
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below Camarilla S3 or 4h trend changes to down
            if close[i] < s3_4h_aligned[i] or not trending_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price rises back above Camarilla R3 or 4h trend changes to up
            if close[i] > r3_4h_aligned[i] or not trending_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: On 1h timeframe, price breaking above/below 4h Camarilla R3/S3 levels with volume surge confirmation and 4h EMA34 trend filter captures institutional breakout momentum. 
# The 4h timeframe provides the primary signal direction, while 1h is used for precise entry timing. 
# Camarilla R3/S3 represent stronger support/resistance levels, reducing false breakouts. 
# Volume surge filter (2.0x 12-period average) confirms institutional participation. 
# Cooldown period (4 bars) prevents overtrading. 
# Position size fixed at 0.20 (20%) to balance risk and reward while minimizing fee churn. 
# Target: 60-150 total trades over 4 years (15-37/year) to stay within acceptable fee drag limits. 
# Works in both bull and bear markets by following the 4h trend direction for breakouts.