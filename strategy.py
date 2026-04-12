#!/usr/bin/env python3
"""
6h_1d_ElderRay_Plus
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA(50) trend filter and volume confirmation. 
Elder Ray > 0 indicates bullish power; < 0 indicates bearish power. We enter when power aligns with 1d trend and volume confirms. 
Exit when power reverses or volume drops. Designed for low trade frequency (~20-40/year) by requiring strong directional alignment.
Works in bull/bear via 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ElderRay_Plus"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY EMA(50) FOR TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    if len(close_1d) >= 50:
        ema_50_1d = np.zeros_like(close_1d)
        ema_50_1d[0] = close_1d[0]
        alpha = 2.0 / (50 + 1)
        for i in range(1, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    else:
        ema_50_1d = np.full_like(close_1d, np.nan)
    
    # Align daily EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === ELDER RAY INDEX (6h) ===
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    # We'll use EMA(13) as the reference
    if len(close) >= 13:
        ema_13 = np.zeros_like(close)
        ema_13[0] = close[0]
        alpha = 2.0 / (13 + 1)
        for i in range(1, len(close)):
            ema_13[i] = alpha * close[i] + (1 - alpha) * ema_13[i-1]
    else:
        ema_13 = np.full_like(close, np.nan)
    
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume average (20-period for 6h = ~5 days) for confirmation
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: price above/below 1d EMA(50)
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Elder Ray signals
        strong_bull = bull_power[i] > 0 and bear_power[i] < 0  # Both bulls and bears agree on direction
        strong_bear = bear_power[i] < 0 and bull_power[i] > 0  # Same condition, will be used differently
        
        # Actually: Bull Power > 0 = bullish, Bear Power < 0 = bearish
        bullish_power = bull_power[i] > 0
        bearish_power = bear_power[i] < 0
        
        # Long when bullish power aligns with uptrend and volume confirms
        long_setup = bullish_power and price_above_ema and vol_confirm
        # Short when bearish power aligns with downtrend and volume confirms
        short_setup = bearish_power and price_below_ema and vol_confirm
        
        # Exit when power diverges or volume weakens
        exit_long = not bullish_power or not vol_confirm
        exit_short = not bearish_power or not vol_confirm
        
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