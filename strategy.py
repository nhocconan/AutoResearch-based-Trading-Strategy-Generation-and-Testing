#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Long when: Williams %R(14) < -80 (oversold) AND 1d EMA50 trend up (close > EMA50) AND volume > 1.5x 20-period MA
# Short when: Williams %R(14) > -20 (overbought) AND 1d EMA50 trend down (close < EMA50) AND volume > 1.5x 20-period MA
# Exit when: Williams %R reverts to midpoint (-50) OR volume drops below average
# Uses Williams %R for extreme reversals, 1d EMA50 for trend alignment, volume for conviction
# Timeframe: 4h, HTF: 1d for EMA50. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_WilliamsR_1dEMA50_VolumeConfirm"
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
    
    # Williams %R(14) on 4h
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    else:
        williams_r = np.full(n, np.nan)
    
    # Williams %R signals
    oversold = williams_r < -80
    overbought = williams_r > -20
    exit_signal = np.abs(williams_r + 50) < 5  # near midpoint -50
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
        volume_exit = volume < vol_ma_20  # exit on low volume
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_exit = np.ones(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d
    close_1d = df_1d['close'].values
    if len(close_1d) >= 50:
        ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50 = np.full(len(df_1d), np.nan)
    
    # 1d trend: close > EMA50 = uptrend, close < EMA50 = downtrend
    trend_up = close_1d > ema_50
    trend_down = close_1d < ema_50
    
    # Align 1d trend to 4h timeframe
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up.astype(float))
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: oversold + uptrend + volume confirmation
            if (oversold[i] and 
                trend_up_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: overbought + downtrend + volume confirmation
            elif (overbought[i] and 
                  trend_down_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R reverts to midpoint OR low volume
            if (exit_signal[i] or volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R reverts to midpoint OR low volume
            if (exit_signal[i] or volume_exit[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals