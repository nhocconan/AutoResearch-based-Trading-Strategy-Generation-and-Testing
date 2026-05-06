#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 12h Donchian upper band AND 1d close > 1d EMA34 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below 12h Donchian lower band AND 1d close < 1d EMA34 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the 12h Donchian middle band (20-period average of high/low)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Donchian channels provide robust price structure that works in both bull and bear markets
# 1d EMA34 filters for higher timeframe trend alignment to avoid counter-trend trades
# Volume spike confirmation reduces false breakouts during low participation

name = "12h_Donchian20_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels and 1d EMA34 ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 20 or len(df_1d) < 34:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_ub = high_20  # Upper band
    donchian_lb = low_20   # Lower band
    donchian_mb = (high_20 + low_20) / 2.0  # Middle band (exit level)
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed bars)
    donchian_ub_aligned = align_htf_to_ltf(prices, df_12h, donchian_ub)
    donchian_lb_aligned = align_htf_to_ltf(prices, df_12h, donchian_lb)
    donchian_mb_aligned = align_htf_to_ltf(prices, df_12h, donchian_mb)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_ub_aligned[i]) or np.isnan(donchian_lb_aligned[i]) or 
            np.isnan(donchian_mb_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper band AND uptrend AND volume spike
            if close[i] > donchian_ub_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower band AND downtrend AND volume spike
            elif close[i] < donchian_lb_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests middle band from above
            if close[i] <= donchian_mb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests middle band from below
            if close[i] >= donchian_mb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals