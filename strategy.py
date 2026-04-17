# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
1d_WeeklyChannel_Trend_Scalper_v1
Weekly Donchian channel breakout with intraday pullback entries on 1d timeframe.
Uses weekly high/low channel for trend direction, daily RSI for entry timing,
and volume confirmation to filter false breakouts. Designed for low-frequency
trading to minimize fee drag in both bull and bear markets.
Target: 20-60 total trades over 4 years (5-15/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Donchian Channel (20-period) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly high and low over 20 periods
    weekly_high = np.full_like(high_1w, np.nan)
    weekly_low = np.full_like(low_1w, np.nan)
    
    for i in range(len(high_1w)):
        if i >= 19:
            weekly_high[i] = np.max(high_1w[i-19:i+1])
            weekly_low[i] = np.min(low_1w[i-19:i+1])
        else:
            weekly_high[i] = np.nan
            weekly_low[i] = np.nan
    
    # Align weekly channel to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # === Daily RSI (14-period) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    for i in range(len(close)):
        if i >= 14:
            if i == 14:
                avg_gain[i] = np.mean(gain[1:15])
                avg_loss[i] = np.mean(loss[1:15])
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Daily Volume Confirmation (20-period average) ===
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(len(volume)):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.nan
    
    vol_confirm = volume > vol_ma_20 * 1.5  # volume spike: 1.5x average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above weekly high AND RSI not overbought (<70) AND volume confirmation
            if (close[i] > weekly_high_aligned[i] and 
                rsi[i] < 70 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly low AND RSI not oversold (>30) AND volume confirmation
            elif (close[i] < weekly_low_aligned[i] and 
                  rsi[i] > 30 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below weekly low OR RSI oversold (<30)
            if (close[i] < weekly_low_aligned[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly high OR RSI overbought (>70)
            if (close[i] > weekly_high_aligned[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyChannel_Trend_Scalper_v1"
timeframe = "1d"
leverage = 1.0