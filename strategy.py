#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND 12h close > 12h EMA50 AND volume > 1.5 * 20-period volume MA
# Short when price breaks below Donchian lower band AND 12h close < 12h EMA50 AND volume > 1.5 * 20-period volume MA
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-40 trades/year per symbol.
# Donchian provides structure; 12h EMA50 filters trend to avoid counter-trend trades; volume confirmation ensures breakout strength.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Uses 12h for HTF trend as specified in experiment parameters.

name = "4h_Donchian20_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian bands (20-period) based on previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper band: highest high of previous 20 bars
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low of previous 20 bars
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Shift to use previous bar's levels (breakout of previous bar's Donchian)
    donchian_high = np.roll(donchian_high, 1)
    donchian_low = np.roll(donchian_low, 1)
    donchian_high[0] = np.nan  # First value invalid after roll
    donchian_low[0] = np.nan
    
    # Align Donchian bands to prices timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_12h = close_12h > ema_50_12h
    downtrend_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 4h timeframe
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    # Calculate volume confirmation: volume > 1.5 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(uptrend_12h_aligned[i]) or np.isnan(downtrend_12h_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Donchian upper AND 12h uptrend AND volume confirmation
            if (close[i] > donchian_high_aligned[i] and 
                uptrend_12h_aligned[i] > 0.5 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Donchian lower AND 12h downtrend AND volume confirmation
            elif (close[i] < donchian_low_aligned[i] and 
                  downtrend_12h_aligned[i] > 0.5 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Donchian lower OR 12h trend changes to downtrend
            if (close[i] < donchian_low_aligned[i] or 
                downtrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Donchian upper OR 12h trend changes to uptrend
            if (close[i] > donchian_high_aligned[i] or 
                uptrend_12h_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals