#!/usr/bin/env python3
"""
1d_Keltner_Channel_Breakout_WeeklyTrend_VolumeConfirm
Hypothesis: 1d price breaks above/below Keltner Channel (EMA20 ± 2*ATR10) with weekly EMA50 trend filter and volume confirmation (>1.5x 20-day avg).
Long when price > upper KC + weekly EMA50 uptrend + volume spike.
Short when price < lower KC + weekly EMA50 downtrend + volume spike.
Exit on opposite KC touch or trend reversal.
Uses discrete sizing (0.25) to limit drawdown in 2022 crash while capturing trends.
Targets 7-25 trades/year (30-100 total over 4 years) to minimize fee drag.
Works in bull via trend-following breakouts, in bear via mean reversion at channel extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner Channel calculations (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA20 and ATR10 for 1d
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    atr_10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel: EMA20 ± 2*ATR10
    upper_kc_1d = ema_20_1d + (2.0 * atr_10_1d)
    lower_kc_1d = ema_20_1d - (2.0 * atr_10_1d)
    
    # Align KC levels to original timeframe
    upper_kc_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_kc_1d)
    lower_kc_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_kc_1d)
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_kc_1d_aligned[i]) or np.isnan(lower_kc_1d_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above upper KC with weekly uptrend and volume spike
            long_signal = (close[i] > upper_kc_1d_aligned[i]) and (close[i] > ema_50_1w_aligned[i]) and vol_spike[i]
            # Short: price breaks below lower KC with weekly downtrend and volume spike
            short_signal = (close[i] < lower_kc_1d_aligned[i]) and (close[i] < ema_50_1w_aligned[i]) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: price touches lower KC or weekly trend reverses
            exit_signal = (close[i] < lower_kc_1d_aligned[i]) or (close[i] < ema_50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price touches upper KC or weekly trend reverses
            exit_signal = (close[i] > upper_kc_1d_aligned[i]) or (close[i] > ema_50_1w_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Keltner_Channel_Breakout_WeeklyTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0