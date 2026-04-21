#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume confirmation.
# Uses 1d EMA for trend direction to avoid counter-trend trades. Volume > 1.5x average confirms breakout strength.
# Donchian channels provide clear breakout levels. Target: 12-37 trades/year per symbol.
# Position size: 0.25 to manage risk during drawdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA (50-period) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12-hour Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Vectorized rolling max/min
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation using 12h volume
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (12h close and 12h volume)
        price_close = prices['close'].iloc[i]
        vol_12h_current = align_htf_to_ltf(prices, df_12h, vol_12h)[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high + volume surge + price > 1d EMA (uptrend)
            if (price_close > donchian_high_aligned[i] and
                vol_12h_current > 1.5 * vol_ma_20_12h_aligned[i] and
                price_close > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low + volume surge + price < 1d EMA (downtrend)
            elif (price_close < donchian_low_aligned[i] and
                  vol_12h_current > 1.5 * vol_ma_20_12h_aligned[i] and
                  price_close < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite Donchian level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price < Donchian low or trend turns down
                if (price_close < donchian_low_aligned[i]) or (price_close < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price > Donchian high or trend turns up
                if (price_close > donchian_high_aligned[i]) or (price_close > ema_50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Volume_Trend"
timeframe = "12h"
leverage = 1.0