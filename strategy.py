#!/usr/bin/env python3
"""
Hypothesis: 1-day Keltner Channel breakout with 1-week EMA trend filter and volume confirmation.
Long when price > Upper KC (20,2) and weekly EMA(50) rising + volume > 20-day avg volume.
Short when price < Lower KC (20,2) and weekly EMA(50) falling + volume > 20-day avg volume.
Exit when price crosses middle line or volume drops below average.
Keltner adapts to volatility; weekly EMA filters trend direction; volume ensures institutional interest.
Works in bull/bear by following institutional volume with volatility-adjusted breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_slope = np.diff(ema_50_1w, prepend=ema_50_1w[0])
    ema_50_1w_slope = np.append(ema_50_1w_slope, ema_50_1w_slope[-1])  # same length
    ema_50_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_slope)
    
    # Load 1-day data for ATR (KC width) and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # ATR(20) for KC width
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    
    # 20-day average volume for filter
    volume_1d = df_1d['volume'].values
    avg_vol_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    # Keltner Channel (20,2) on daily
    kc_middle = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = kc_middle + 2 * atr_20
    kc_lower = kc_middle - 2 * atr_20
    kc_middle_aligned = align_htf_to_ltf(prices, df_1d, kc_middle)
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if np.isnan(kc_middle_aligned[i]) or np.isnan(ema_50_1w_slope_aligned[i]) or np.isnan(avg_vol_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper KC, weekly EMA rising, volume above average
            if (high[i] > kc_upper_aligned[i] and 
                ema_50_1w_slope_aligned[i] > 0 and 
                volume[i] > avg_vol_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower KC, weekly EMA falling, volume above average
            elif (low[i] < kc_lower_aligned[i] and 
                  ema_50_1w_slope_aligned[i] < 0 and 
                  volume[i] > avg_vol_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below middle KC
                if close[i] < kc_middle_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above middle KC
                if close[i] > kc_middle_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_KC_Breakout_1wEMA_Volume"
timeframe = "1d"
leverage = 1.0