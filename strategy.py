#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Elder Ray Index (Bull/Bear Power) with 1w EMA trend filter and volume confirmation
# Long when 1d Bull Power > 0 AND 1w EMA34 > EMA89 AND volume > 1.5 * avg_volume(20)
# Short when 1d Bear Power < 0 AND 1w EMA34 < EMA89 AND volume > 1.5 * avg_volume(20)
# Exit when Elder Power reverses sign or touches zero
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# 1d Elder Ray measures bull/bear strength relative to 13-period EMA, effective in both bull and bear markets
# 1w EMA filter ensures alignment with weekly trend, reducing counter-trend trades
# Volume confirmation filters weak signals
# Works in bull (strong bull power with uptrend) and bear (strong bear power with downtrend)

name = "6h_1dElderRay_1wEMATrend_Volume"
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
    
    # Get 1d data ONCE before loop for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for EMA13 and calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    close_series_1d = pd.Series(close_1d)
    ema13_1d = close_series_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align 1d Elder Ray to 6h timeframe (wait for completed 1d bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 and EMA89
    close_series_1w = pd.Series(close_1w)
    ema_34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1w = close_series_1w.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1w EMA values to 6h timeframe (wait for completed 1w bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_89_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d Bull Power > 0 (bulls in control) with 1w EMA34 > EMA89 and volume confirmation
            if (bull_power_aligned[i] > 0 and 
                ema_34_aligned[i] > ema_89_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1d Bear Power < 0 (bears in control) with 1w EMA34 < EMA89 and volume confirmation
            elif (bear_power_aligned[i] < 0 and 
                  ema_34_aligned[i] < ema_89_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 1d Bull Power <= 0 (bulls losing control) or Bear Power >= 0 (bears taking over)
            if bull_power_aligned[i] <= 0 or bear_power_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 1d Bear Power >= 0 (bears losing control) or Bull Power <= 0 (bulls taking over)
            if bear_power_aligned[i] >= 0 or bull_power_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals