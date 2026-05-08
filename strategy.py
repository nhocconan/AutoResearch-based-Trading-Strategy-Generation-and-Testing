#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d EMA(34) as trend filter, 4h Donchian(20) breakout, and volume confirmation.
# Long when 1d EMA > price (bullish trend), price breaks above 4h Donchian upper band, volume > 1.5x average.
# Short when 1d EMA < price (bearish trend), price breaks below 4h Donchian lower band, volume > 1.5x average.
# Fixed position size of 0.25 to limit overtrading and manage drawdown. Target: 20-50 trades/year.
# Designed to work in bull (trend follow) and bear (trend still exists in downtrends) by using daily trend filter.

name = "4h_1dEMA34_4hDonchian_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Get 4h data for Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 1-day EMA(34)
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_bullish = close > ema_1d[-1] if len(ema_1d) > 0 else False  # Will be replaced by aligned version
    
    # 4h Donchian(20) bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 1-day EMA to 4h (no extra delay needed for EMA as it's based on closed daily bar)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_bullish_aligned = ema_1d_aligned < close  # Bullish when price above daily EMA
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: daily EMA bullish (price > EMA), price breaks above 4h Donchian upper band, volume spike
            if (ema_bullish_aligned[i] and
                close[i] > donchian_high[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: daily EMA bearish (price < EMA), price breaks below 4h Donchian lower band, volume spike
            elif (not ema_bullish_aligned[i] and
                  close[i] < donchian_low[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: trend flip, price breaks below Donchian lower band, or max 20 bars held
            if (not ema_bullish_aligned[i] or 
                close[i] < donchian_low[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend flip, price breaks above Donchian upper band, or max 20 bars held
            if (ema_bullish_aligned[i] or 
                close[i] > donchian_high[i] or
                i - entry_bar >= 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals