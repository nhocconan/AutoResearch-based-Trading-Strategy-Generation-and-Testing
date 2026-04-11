#!/usr/bin/env python3
# 12h_1w_donchian_breakout_v1
# Strategy: 12-hour Donchian breakout with 1-week trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Price breaks out of weekly Donchian channels during trending markets,
# with volume confirmation to avoid false breakouts. Works in bull by catching breakouts
# above weekly highs and in bear by catching breakdowns below weekly lows. Uses volume
# spike to ensure institutional participation and reduce whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 1w high and low for Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-period rolling max/min for Donchian bands
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to 12h timeframe (wait for weekly close)
    high_max_20_aligned = align_htf_to_ltf(prices, df_1w, high_max_20)
    low_min_20_aligned = align_htf_to_ltf(prices, df_1w, low_min_20)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Breakout signals: price breaks weekly Donchian bands with volume
        long_signal = (price_high > high_max_20_aligned[i]) and vol_spike[i] and uptrend_1d
        short_signal = (price_low < low_min_20_aligned[i]) and vol_spike[i] and downtrend_1d
        
        # Exit when price returns to opposite Donchian band or trend reverses
        exit_long = position == 1 and (price_low < low_min_20_aligned[i] or not uptrend_1d)
        exit_short = position == -1 and (price_high > high_max_20_aligned[i] or not downtrend_1d)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals