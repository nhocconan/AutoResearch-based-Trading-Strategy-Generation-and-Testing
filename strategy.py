#!/usr/bin/env python3
"""
1d_KAMA_Direction_1wEMA34_Trend_VolumeConfirm
Hypothesis: On daily timeframe, Kaufman Adaptive Moving Average (KAMA) direction with 1-week EMA34 trend filter and volume confirmation.
KAMA adapts to market noise - fast in trends, slow in ranges, reducing whipsaws. 1-week EMA34 ensures alignment with weekly momentum.
Volume confirmation filters low-conviction moves. Works in bull (trend following) and bear (mean reversion at extremes) markets.
Target: 15-30 trades/year to stay within proven winning range for 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_fast=2, er_slow=30):
    """Calculate Kaufman Adaptive Moving Average with min_periods"""
    if len(close) < er_slow:
        return np.full_like(close, np.nan, dtype=float)
    close_series = pd.Series(close)
    change = abs(close_series.diff(er_slow))
    volatility = close_series.diff().abs().rolling(window=er_slow, min_periods=er_slow).sum()
    er = (change / volatility.replace(0, np.nan)).fillna(0)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for EMA34 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA34 trend filter
    ema_34_1w = calculate_ema(df_1w['close'].values, 34)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d KAMA direction (primary timeframe)
    kama = calculate_kama(close, er_fast=2, er_slow=30)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for KAMA (30) + EMA (34) + volume MA (20)
    start_idx = max(30, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals - require: KAMA direction + 1w EMA34 trend alignment + volume confirmation
            kama_up = kama[i] > kama[i-1]
            kama_down = kama[i] < kama[i-1]
            
            long_entry = (kama_up and 
                         (curr_close > ema_34_1w_aligned[i]) and 
                         volume_confirm[i])
            short_entry = (kama_down and 
                          (curr_close < ema_34_1w_aligned[i]) and 
                          volume_confirm[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when KAMA turns down OR price closes below 1w EMA34
            if (kama[i] < kama[i-1]) or (curr_close < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when KAMA turns up OR price closes above 1w EMA34
            if (kama[i] > kama[i-1]) or (curr_close > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_1wEMA34_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0