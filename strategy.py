#!/usr/bin/env python3
"""
4h_1d_Donchian_Breakout_Volume_Regime_v1
Hypothesis: On 4h timeframe, buy breakouts above Donchian(20) high with 1d trend filter and volume confirmation,
sell breakdowns below Donchian(20) low with 1d downtrend and volume confirmation. Exit at opposite Donchian level.
Uses daily volatility regime filter to avoid choppy markets. Designed for low trade frequency
(20-50/year) by requiring multiple confluence factors. Works in bull/bear via 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian_Breakout_Volume_Regime_v1"
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
    
    # === DAILY TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily EMA(50) for trend direction
    ema_50 = np.zeros_like(close_1d)
    ema_sum = 0.0
    ema_count = 0
    for i in range(len(close_1d)):
        ema_sum += close_1d[i]
        ema_count += 1
        if i >= 50:
            ema_sum -= close_1d[i-50]
            ema_count -= 1
        if ema_count > 0:
            ema_50[i] = ema_sum / ema_count
        else:
            ema_50[i] = 0.0
    
    # Trend: price above EMA50 = uptrend, below = downtrend
    trend_up = close_1d > ema_50
    trend_down = close_1d < ema_50
    
    # === DAILY VOLATILITY REGIME FILTER ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Daily ATR(14) for volatility measurement
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = np.zeros_like(tr)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14[i] = np.nanmean(tr[1:i+1])
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    # Volatility regime: low volatility = trending market
    vol_ma = np.zeros_like(atr_14)
    for i in range(len(atr_14)):
        if i >= 30:
            vol_ma[i] = np.mean(atr_14[i-29:i+1])
        else:
            vol_ma[i] = np.nan
    # Low volatility regime (trending) when current ATR < MA
    vol_regime = atr_14 < vol_ma
    
    # === 4H DONCHIAN CHANNEL (20-period) ===
    # Donchian high: highest high of last 20 periods
    donch_high = np.full(n, np.nan)
    # Donchian low: lowest low of last 20 periods
    donch_low = np.full(n, np.nan)
    
    high_sum = 0.0
    high_count = 0
    low_sum = 0.0
    low_count = 0
    # We'll use simple rolling max/min via deque-like approach but optimized
    for i in range(n):
        # Add current values
        high_sum += high[i]
        low_sum += low[i]
        high_count += 1
        low_count += 1
        # Remove values outside 20-period window
        if i >= 20:
            high_sum -= high[i-20]
            low_sum -= low[i-20]
            high_count -= 1
            low_count -= 1
        # Calculate for window >= 20
        if i >= 19 and high_count == 20:
            donch_high[i] = high_sum / 20  # This is average, not max - need to fix
        if i >= 19 and low_count == 20:
            donch_low[i] = low_sum / 20   # This is average, not min - need to fix
    
    # Fix: Proper Donchian calculation using running max/min
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    # Initialize deques for tracking max/min
    from collections import deque
    high_window = deque(maxlen=20)
    low_window = deque(maxlen=20)
    
    for i in range(n):
        high_window.append(high[i])
        low_window.append(low[i])
        if len(high_window) == 20:
            donch_high[i] = max(high_window)
            donch_low[i] = min(low_window)
    
    # Align data to 4h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    # Volume average (20-period for 4h = ~10 hours) for confirmation
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
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or 
            np.isnan(vol_regime_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Only trade in low volatility (trending) regime
        in_trend_regime = vol_regime_aligned[i] > 0.5
        
        # Entry conditions
        long_setup = (close[i] > donch_high[i]) and trend_up_aligned[i] > 0.5 and vol_confirm and in_trend_regime
        short_setup = (close[i] < donch_low[i]) and trend_down_aligned[i] > 0.5 and vol_confirm and in_trend_regime
        
        # Exit conditions: opposite Donchian level
        exit_long = close[i] < donch_low[i]
        exit_short = close[i] > donch_high[i]
        
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