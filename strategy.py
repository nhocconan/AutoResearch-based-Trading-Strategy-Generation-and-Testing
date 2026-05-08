#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week trend filter (EMA34) and 1-day Donchian breakout with volume confirmation.
# Long when 1w EMA > price (bullish trend), price breaks above 1d Donchian upper band, volume > 2x average.
# Short when 1w EMA < price (bearish trend), price breaks below 1d Donchian lower band, volume > 2x average.
# Exit on trend reversal or Donchian break in opposite direction.
# Uses position size 0.25 to balance return and drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Designed to capture trends in both bull and bear markets by using 1w trend filter, with volume to confirm breakout strength.

name = "1d_1wEMA34_1dDonchian_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 1d data for Donchian bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1-week EMA(34)
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1-day Donchian(20) bands
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1w EMA bullish (price > EMA), price breaks above 1d Donchian upper band, volume spike
            if (close[i] > ema_1w_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: 1w EMA bearish (price < EMA), price breaks below 1d Donchian lower band, volume spike
            elif (close[i] < ema_1w_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: trend reversal or price breaks below Donchian lower band
            if (close[i] < ema_1w_aligned[i] or 
                close[i] < donchian_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or price breaks above Donchian upper band
            if (close[i] > ema_1w_aligned[i] or 
                close[i] > donchian_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals