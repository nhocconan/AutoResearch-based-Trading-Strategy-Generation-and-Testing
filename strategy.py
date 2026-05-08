#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly EMA(34) as trend filter, 1d Donchian(20) breakout, and volume confirmation.
# Long when weekly EMA > price (bullish), price breaks above 1d Donchian upper band, volume > 1.5x average.
# Short when weekly EMA < price (bearish), price breaks below 1d Donchian lower band, volume > 1.5x average.
# Fixed position size of 0.25 to limit risk and trade frequency.
# Target: 20-60 total trades over 4 years (5-15/year) to minimize fee drag.
# Works in bull (trend follow) and bear (trend still exists in downtrends).

name = "1d_weeklyEMA34_1dDonchian_Volume"
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
    
    # Get weekly data for EMA trend filter
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
    
    # Weekly EMA(34)
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_bullish = ema_34 > close_1w  # EMA above price = bullish
    
    # 1d Donchian(20) bands
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align weekly EMA to 1d
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish.astype(float))
    
    # Align 1d Donchian bands to 1d
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 54  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly EMA bullish, price breaks above 1d Donchian upper band, volume spike
            if (ema_bullish_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: weekly EMA bearish, price breaks below 1d Donchian lower band, volume spike
            elif (not ema_bullish_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: EMA flip, price breaks below Donchian lower band, or max 10 days held
            if (not ema_bullish_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 10):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: EMA flip, price breaks above Donchian upper band, or max 10 days held
            if (ema_bullish_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 10):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals