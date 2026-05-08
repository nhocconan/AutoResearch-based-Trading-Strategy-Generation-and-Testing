#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly EMA(21) as trend filter, 6h Donchian(15) breakout, and volume confirmation.
# Long when weekly EMA > price (bullish), price breaks above 6h Donchian upper band, volume > 1.8x average.
# Short when weekly EMA < price (bearish), price breaks below 6h Donchian lower band, volume > 1.8x average.
# Exit on trend reversal, Donchian break in opposite direction, or max 30 bars held.
# Weekly EMA provides strong trend filter for multi-week trends, Donchian captures breakouts, volume confirms strength.
# Designed for 6h timeframe to capture multi-week moves in both bull and bear markets with low trade frequency.

name = "6h_WeeklyEMA21_6hDonchian_Volume"
timeframe = "6h"
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
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 6h data for Donchian bands
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 15:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Weekly EMA(21)
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 6-hour Donchian(15) bands
    donchian_high = pd.Series(high_6h).rolling(window=15, min_periods=15).max().values
    donchian_low = pd.Series(low_6h).rolling(window=15, min_periods=15).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
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
            # Long: weekly EMA bullish (price > EMA), price breaks above 6h Donchian upper band, volume spike
            if (close[i] > ema_1w_aligned[i] and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.8):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: weekly EMA bearish (price < EMA), price breaks below 6h Donchian lower band, volume spike
            elif (close[i] < ema_1w_aligned[i] and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.8):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: trend reversal, price breaks below Donchian lower band, or max 30 bars held
            if (close[i] < ema_1w_aligned[i] or 
                close[i] < donchian_low_aligned[i] or
                i - entry_bar >= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal, price breaks above Donchian upper band, or max 30 bars held
            if (close[i] > ema_1w_aligned[i] or 
                close[i] > donchian_high_aligned[i] or
                i - entry_bar >= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals