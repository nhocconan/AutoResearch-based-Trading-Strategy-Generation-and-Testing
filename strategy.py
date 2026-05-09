#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 Breakout + 1d Trend + Volume Spike
# Camarilla R4/S4 are extreme levels that rarely break, so when they do with volume and trend alignment,
# it signals strong institutional momentum. Works in both bull (breakouts continue) and bear (sharp reversals).
# Target: 25-40 trades/year (100-160 over 4 years) to avoid fee drag.
name = "4h_Camarilla_R4S4_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation and trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC (avoid look-ahead)
    daily_high = df_daily['high'].shift(1).values
    daily_low = df_daily['low'].shift(1).values
    daily_close = df_daily['close'].shift(1).values
    
    # Camarilla equations
    spread = daily_high - daily_low
    camarilla_pivot = (daily_high + daily_low + daily_close) / 3.0
    r4 = camarilla_pivot + (1.1 * spread / 2)
    s4 = camarilla_pivot - (1.1 * spread / 2)
    r3 = camarilla_pivot + (1.1 * spread / 4)
    s3 = camarilla_pivot - (1.1 * spread / 4)
    r2 = camarilla_pivot + (1.1 * spread / 6)
    s2 = camarilla_pivot - (1.1 * spread / 6)
    r1 = camarilla_pivot + (1.1 * spread / 12)
    s1 = camarilla_pivot - (1.1 * spread / 12)
    
    # Align daily Camarilla levels to 4h
    r4_4h = align_htf_to_ltf(prices, df_daily, r4)
    s4_4h = align_htf_to_ltf(prices, df_daily, s4)
    r3_4h = align_htf_to_ltf(prices, df_daily, r3)
    s3_4h = align_htf_to_ltf(prices, df_daily, s3)
    
    # Daily EMA50 for trend filter
    ema50_daily = pd.Series(df_daily['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_4h = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or np.isnan(r3_4h[i]) or 
            np.isnan(s3_4h[i]) or np.isnan(ema50_daily_4h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above R4 with daily uptrend and volume spike
            if close[i] > r4_4h[i] and close[i] > ema50_daily_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below S4 with daily downtrend and volume spike
            elif close[i] < s4_4h[i] and close[i] < ema50_daily_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below R3 OR daily trend turns down
            if close[i] < r3_4h[i] or close[i] < ema50_daily_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above S3 OR daily trend turns up
            if close[i] > s3_4h[i] or close[i] > ema50_daily_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals