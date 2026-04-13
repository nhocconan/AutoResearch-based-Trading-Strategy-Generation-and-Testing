# 12h Donchian breakout with 1w trend filter and volume confirmation
# Uses 12h Donchian breakouts for entries, 1w EMA for trend filter, and volume confirmation
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag
# Works in bull (breakouts with trend) and bear (mean reversion at extremes) via volatility filter
# Discrete position sizing: 0.0, ±0.25 to reduce churn

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = np.full(len(high_12h), np.nan)
    donchian_low = np.full(len(low_12h), np.nan)
    for i in range(20, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-20:i])
        donchian_low[i] = np.min(low_12h[i-20:i])
    
    # Calculate 20-period average volume on 12h
    avg_volume_12h = np.full(len(vol_12h), np.nan)
    for i in range(20, len(vol_12h)):
        avg_volume_12h[i] = np.mean(vol_12h[i-20:i])
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 50-period EMA on weekly close
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = close_1w[:50].mean()
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * 0.0392) + (ema_50_1w[i-1] * 0.9608)
    
    # Align all indicators to 12h timeframe (our trading timeframe)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(avg_volume_12h_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > average volume
        vol_confirm = volume[i] > avg_volume_12h_aligned[i]
        
        # Donchian breakout conditions
        donchian_breakout_long = close[i] > donchian_high_aligned[i]
        donchian_breakout_short = close[i] < donchian_low_aligned[i]
        
        # Trend filter: price above/below 50-week EMA
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions: breakout with trend and volume
        long_entry = donchian_breakout_long and uptrend and vol_confirm
        short_entry = donchian_breakout_short and downtrend and vol_confirm
        
        # Exit conditions: opposite breakout or loss of trend
        exit_long = position == 1 and (donchian_breakout_short or not uptrend)
        exit_short = position == -1 and (donchian_breakout_long or not downtrend)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_donchian_trend_volume"
timeframe = "12h"
leverage = 1.0