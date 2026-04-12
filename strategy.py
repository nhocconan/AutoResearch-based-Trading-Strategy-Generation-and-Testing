#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Trend_v1
Hypothesis: In both bull and bear markets, price breaks Camarilla H3/L3 levels with 1-day trend (EMA50) and volume confirmation. 
Long when price > H3 and close > EMA50; short when price < L3 and close < EMA50. 
Uses 4h for entry timing and 1d for Camarilla levels and trend filter. Target 20-40 trades/year.
Works in bull (breakout continuation) and bear (mean reversion failure at H3/L3).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # === 1D CAMARILLA LEVELS (based on previous day) ===
    # H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    prev_close = np.roll(daily_close, 1)
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    # First day: use same day's values (no look-ahead as we use previous day's data)
    prev_close[0] = daily_close[0]
    prev_high[0] = daily_high[0]
    prev_low[0] = daily_low[0]
    
    cam_h3 = prev_close + 1.1 * (prev_high - prev_low)
    cam_l3 = prev_close - 1.1 * (prev_high - prev_low)
    
    cam_h3_4h = align_htf_to_ltf(prices, df_1d, cam_h3)
    cam_l3_4h = align_htf_to_ltf(prices, df_1d, cam_l3)
    
    # === 1D EMA50 TREND FILTER ===
    ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50)
    
    # === VOLUME SPIKE (1.5x) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any data invalid
        if (np.isnan(cam_h3_4h[i]) or np.isnan(cam_l3_4h[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with trend filter
        breakout_up = high[i] > cam_h3_4h[i]
        breakout_down = low[i] < cam_l3_4h[i]
        trend_up = close[i] > ema50_4h[i]
        trend_down = close[i] < ema50_4h[i]
        
        # Entry conditions
        long_entry = breakout_up and trend_up and vol_spike[i]
        short_entry = breakout_down and trend_down and vol_spike[i]
        
        # Exit: opposite breakout or trend fails
        long_exit = not breakout_up or not trend_up
        short_exit = not breakout_down or not trend_down
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals