#/usr/bin/env python3
"""
4h_1d_4h_breakout_v1
Hypothesis: 4h price breaking above/below daily Donchian(20) high/low with 4h EMA(21) trend filter and volume confirmation (1.5x). Exits on opposite Donchian break. Designed for low trade frequency (<30/year) by requiring strong breakouts, trend alignment, and volume surge. Works in bull/bear via EMA trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_4h_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DONCHIAN(20) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian channels (20-period)
    donch_high_1d = np.full_like(high_1d, np.nan)
    donch_low_1d = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        donch_high_1d[i] = np.max(high_1d[i-20:i])
        donch_low_1d[i] = np.min(low_1d[i-20:i])
    
    # === 4H EMA(21) FOR TREND FILTER ===
    if len(close) >= 21:
        ema_21 = np.zeros_like(close)
        ema_21[0] = close[0]
        alpha = 2.0 / (21 + 1)
        for i in range(1, len(close)):
            ema_21[i] = alpha * close[i] + (1 - alpha) * ema_21[i-1]
    else:
        ema_21 = np.full_like(close, np.nan)
    
    # Align daily Donchian to 4h timeframe
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # Volume average (20-period for 4h = ~1.3 days) for confirmation
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
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or 
            np.isnan(ema_21[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: price above/below 4h EMA(21)
        price_above_ema = close[i] > ema_21[i]
        price_below_ema = close[i] < ema_21[i]
        
        # Breakout entries at daily Donchian with volume and trend filters
        long_setup = (close[i] > donch_high_1d_aligned[i]) and vol_confirm and price_above_ema
        short_setup = (close[i] < donch_low_1d_aligned[i]) and vol_confirm and price_below_ema
        
        # Exit on opposite Donchian break
        exit_long = close[i] < donch_low_1d_aligned[i]
        exit_short = close[i] > donch_high_1d_aligned[i]
        
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