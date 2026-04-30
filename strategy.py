#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel, price > 12h EMA50, and volume > 1.8x 20-bar avg.
# Short when price breaks below Donchian lower channel, price < 12h EMA50, and volume > 1.8x 20-bar avg.
# Exit when price reverts to the Donchian middle channel (mean reversion).
# Uses 4h timeframe for balance of trade frequency and signal quality (target: 30-60 trades/year).
# 12h EMA50 provides strong trend alignment; volume confirmation reduces false signals.
# Works in bull markets via breakouts and in bear markets via short breakdowns with trend alignment.
# Target: 120-240 total trades over 4 years.

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_channel = high_rolling_max
    lower_channel = low_rolling_min
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(middle_channel[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_middle = middle_channel[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper channel, price > 12h EMA50, volume spike
            if (curr_close > curr_upper and 
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel, price < 12h EMA50, volume spike
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price reverts to middle channel (mean reversion)
            if curr_close <= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price reverts to middle channel (mean reversion)
            if curr_close >= curr_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals