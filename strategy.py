#!/usr/bin/env python3
"""
4h Donchian Breakout with Daily Trend Filter and Volume Confirmation.
Long when price breaks above 4h Donchian upper band (20-period) and price > daily EMA50.
Short when price breaks below 4h Donchian lower band (20-period) and price < daily EMA50.
Exit when price crosses back to Donchian midpoint (10-period) or opposite band breakout occurs.
Uses volume confirmation (volume > 1.5x 20-period average) to avoid false breakouts.
Position size: 0.25 (25% of capital) to manage drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_daily_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily EMA50 for trend filter (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # === 4h Donchian Channels (20-period high/low) ===
    # Upper band: 20-period high
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    # Middle: 10-period average of high/low for exit
    hl_avg = (high + low) / 2
    donchian_middle = pd.Series(hl_avg).rolling(window=10, min_periods=10).mean().values
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(daily_ema50_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below middle OR breaks lower band (reverse signal)
            if close[i] < donchian_middle[i] or close[i] < donchian_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above middle OR breaks upper band (reverse signal)
            if close[i] > donchian_middle[i] or close[i] > donchian_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: above average volume
            if vol_ratio[i] < 1.5:
                signals[i] = 0.0
                continue
            
            # Long entry: price breaks above upper band AND above daily EMA50 (uptrend)
            if close[i] > donchian_upper[i] and close[i] > daily_ema50_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band AND below daily EMA50 (downtrend)
            elif close[i] < donchian_lower[i] and close[i] < daily_ema50_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals