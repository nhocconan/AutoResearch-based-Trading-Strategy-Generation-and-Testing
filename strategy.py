#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA50 trend filter
# Long when price breaks above Donchian(20) high AND volume > 1.5x 20-bar avg AND price > 12h EMA50
# Short when price breaks below Donchian(20) low AND volume > 1.5x 20-bar avg AND price < 12h EMA50
# Exit when price crosses 12h EMA50 (trend change)
# Uses discrete position sizing (0.25) to reduce fee drag.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to avoid overtrading.
# Donchian channels provide clear structure, volume confirms breakout strength, 12h EMA50 filters trend.
# Works in bull markets via upward breakouts with trend alignment and in bear markets via downward breakouts.

name = "4h_Donchian20_Volume_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-bar average volume (balanced to avoid overtrading)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 12h EMA50 (trend change)
            if curr_close < curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 12h EMA50 (trend change)
            if curr_close > curr_ema50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian(20) high AND volume confirmation AND price > 12h EMA50
            if curr_high > curr_donchian_high and vol_conf and curr_close > curr_ema50_12h:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian(20) low AND volume confirmation AND price < 12h EMA50
            elif curr_low < curr_donchian_low and vol_conf and curr_close < curr_ema50_12h:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals