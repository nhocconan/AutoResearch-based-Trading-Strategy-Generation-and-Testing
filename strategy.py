#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter and volume confirmation.
# In strong trends (price > 1d EMA50), breakouts above/below Donchian(20) have higher probability.
# Volume > 1.5x average confirms breakout strength. Target: 50-150 total trades over 4 years.
# Position size: 0.25 to manage risk during drawdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 6h data for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Upper band: highest high over 20 periods
    upper_band = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    lower_band = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to lower timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_6h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_6h, lower_band)
    
    # Calculate 1-day EMA (50-period) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation using 6h volume
    vol_6h = df_6h['volume'].values
    vol_ma_20_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_6h_current = align_htf_to_ltf(prices, df_6h, vol_6h)[i]
        
        if position == 0:
            # Enter long: price breaks above upper band + volume surge + price > 1d EMA (uptrend)
            if (price_close > upper_band_aligned[i] and
                vol_6h_current > 1.5 * vol_ma_20_6h_aligned[i] and
                price_close > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band + volume surge + price < 1d EMA (downtrend)
            elif (price_close < lower_band_aligned[i] and
                  vol_6h_current > 1.5 * vol_ma_20_6h_aligned[i] and
                  price_close < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to middle of Donchian channel or trend reverses
            exit_signal = False
            mid_band = (upper_band_aligned[i] + lower_band_aligned[i]) / 2
            
            if position == 1:
                # Exit long: price falls below midpoint or trend turns down
                if (price_close < mid_band) or (price_close < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above midpoint or trend turns up
                if (price_close > mid_band) or (price_close > ema_50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_1dEMA50_Volume_Trend"
timeframe = "6h"
leverage = 1.0