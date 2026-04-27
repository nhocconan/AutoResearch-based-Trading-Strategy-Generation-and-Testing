# 1d_WeeklyDonchian20_1dTrend_Volume
# Hypothesis: Weekly Donchian breakouts combined with daily trend filter and volume confirmation
# work across both bull and bear markets by capturing momentum after volatility compression.
# The daily EMA50 ensures we only trade in the direction of the intermediate-term trend,
# reducing false breakouts in ranging markets. Weekly timeframe reduces trade frequency
# to minimize fee drag while capturing significant moves.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly upper band (20-period high)
    donchian_high = np.full(len(high_1w), np.nan)
    for i in range(19, len(high_1w)):
        donchian_high[i] = np.max(high_1w[i-19:i+1])
    
    # Calculate weekly lower band (20-period low)
    donchian_low = np.full(len(low_1w), np.nan)
    for i in range(19, len(low_1w)):
        donchian_low[i] = np.min(low_1w[i-19:i+1])
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50  # EMA50
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with volume and above daily EMA50
            if price > donchian_high_aligned[i] and vol_filter and price > ema_50_1d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below weekly Donchian low with volume and below daily EMA50
            elif price < donchian_low_aligned[i] and vol_filter and price < ema_50_1d_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below weekly Donchian low or trailing stop
            if price < donchian_low_aligned[i] or price < ema_50_1d_aligned[i] - 1.5 * np.abs(price - ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above weekly Donchian high or trailing stop
            if price > donchian_high_aligned[i] or price > ema_50_1d_aligned[i] + 1.5 * np.abs(price - ema_50_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian20_1dTrend_Volume"
timeframe = "1d"
leverage = 1.0