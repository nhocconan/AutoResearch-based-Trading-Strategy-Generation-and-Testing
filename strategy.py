#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d trend filter and 1h Donchian breakout with volume confirmation.
# Long when 4h EMA > price (bullish trend), price breaks above 1h Donchian upper band, volume > 1.8x average.
# Short when 4h EMA < price (bearish trend), price breaks below 1h Donchian lower band, volume > 1.8x average.
# Exit on trend reversal or Donchian break in opposite direction.
# Uses position size 0.20 to manage risk. Target: 60-150 total trades over 4 years (15-37/year).
# Uses 4h/1d for signal direction, 1h only for entry timing.
# Includes session filter (08-20 UTC) to reduce noise.

name = "1h_4hEMA34_1dEMA50_1hDonchian_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get 1h data for Donchian bands
    df_1h = get_htf_data(prices, '1h')  # This is just the prices data itself
    if len(df_1h) < 20:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    
    # 4-hour EMA(34)
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1-day EMA(50)
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-hour Donchian(20) bands
    donchian_high = pd.Series(high_1h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ratio[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h EMA bullish (price > EMA), 1d EMA bullish (price > EMA), price breaks above 1h Donchian upper band, volume spike
            if (close[i] > ema_4h_aligned[i] and
                close[i] > ema_1d_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.20
                position = 1
                entry_bar = i
            # Short: 4h EMA bearish (price < EMA), 1d EMA bearish (price < EMA), price breaks below 1h Donchian lower band, volume spike
            elif (close[i] < ema_4h_aligned[i] and
                  close[i] < ema_1d_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.20
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: trend reversal or price breaks below Donchian lower band
            if (close[i] < ema_4h_aligned[i] or 
                close[i] < ema_1d_aligned[i] or
                close[i] < donchian_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend reversal or price breaks above Donchian upper band
            if (close[i] > ema_4h_aligned[i] or 
                close[i] > ema_1d_aligned[i] or
                close[i] > donchian_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals