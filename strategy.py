#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel AND price > 12h EMA50 AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower channel AND price < 12h EMA50 AND volume > 1.5x 20-bar average.
# Exit when price touches Donchian midpoint (mean of upper and lower) or opposite channel touch.
# Donchian channels provide clear breakout levels with defined risk/reward.
# 12h EMA50 filters for intermediate-term trend to avoid counter-trend entries.
# Volume confirmation ensures breakout validity.
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirmation_v1"
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian Channel (20-period)
    # Upper channel: highest high of last 20 periods
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle channel: average of upper and lower
    middle_channel = (upper_channel + lower_channel) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(middle_channel[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_middle = middle_channel[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above upper channel, uptrend (price > 12h EMA50), volume confirmation
            if (curr_high > curr_upper and 
                curr_close > ema_50_12h_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
            # Short: Break below lower channel, downtrend (price < 12h EMA50), volume confirmation
            elif (curr_low < curr_lower and 
                  curr_close < ema_50_12h_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Price touches middle channel or breaks below lower channel
            if curr_low <= curr_middle or curr_low <= curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit condition: Price touches middle channel or breaks above upper channel
            if curr_high >= curr_middle or curr_high >= curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals