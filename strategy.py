#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above Donchian(20) upper band AND 12h EMA50 > EMA200 (uptrend) AND volume > 1.5x 20-period MA.
# Short when price breaks below Donchian(20) lower band AND 12h EMA50 < EMA200 (downtrend) AND volume > 1.5x 20-period MA.
# Exit when price returns to Donchian(20) middle band or trend reverses.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

name = "4h_Donchian20_12hEMATrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 and EMA200
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Donchian channel (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = (highest_20 + lowest_20) / 2.0
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        trend_up = ema_50_aligned[i] > ema_200_aligned[i]
        trend_down = ema_50_aligned[i] < ema_200_aligned[i]
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: Price breaks above Donchian upper band AND uptrend AND volume spike
            if close_val > highest_20[i] and trend_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band AND downtrend AND volume spike
            elif close_val < lowest_20[i] and trend_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to middle band OR trend reverses to downtrend
            if close_val < middle_20[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to middle band OR trend reverses to uptrend
            if close_val > middle_20[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals