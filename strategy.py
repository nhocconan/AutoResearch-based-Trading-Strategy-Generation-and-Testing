#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with 1d EMA trend filter and volume confirmation.
# In uptrend (price > 1d EMA50), go long on 4h Donchian upper breakout; in downtrend (price < 1d EMA50), go short on 4h Donchian lower breakout.
# Volume confirmation (>1.3x 20-period EMA) reduces false signals. Designed for 1h timeframe targeting 60-150 total trades over 4 years.
# Uses discrete position sizing (0.20) to minimize fee churn and manage drawdown. Works in both bull and bear markets via trend filter.

name = "1h_Donchian_20_1dEMA50_Trend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channel (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Determine trend: price > 1d EMA50 = uptrend, price < 1d EMA50 = downtrend
            if close[i] > ema_50_aligned[i]:
                # Uptrend: long on Donchian upper breakout
                if close[i] > donchian_high_aligned[i] and volume_confirm:
                    signals[i] = 0.20
                    position = 1
            else:
                # Downtrend: short on Donchian lower breakout
                if close[i] < donchian_low_aligned[i] and volume_confirm:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: price retouches Donchian midpoint OR trend reverses OR volume drops
            mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if (close[i] <= mid or 
                close[i] < ema_50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price retouches Donchian midpoint OR trend reverses OR volume drops
            mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if (close[i] >= mid or 
                close[i] > ema_50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals