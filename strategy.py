#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend as trend filter, 4h Donchian(20) breakout, and volume confirmation.
# Long when 12h Supertrend is up, price breaks above 4h Donchian upper band, volume > 1.5x average.
# Short when 12h Supertrend is down, price breaks below 4h Donchian lower band, volume > 1.5x average.
# Uses fixed position size of 0.25 to limit risk and avoid overtrading.
# Designed to work in both bull and bear markets via trend filter and breakout logic.

name = "4h_12hSupertrend_4hDonchian_Volume"
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
    
    # Get 12h data for Supertrend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 4h data for Donchian bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 12h Supertrend (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_12h + low_12h) / 2 + multiplier * atr_12h
    basic_lb = (high_12h + low_12h) / 2 - multiplier * atr_12h
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close_12h)
    final_lb = np.zeros_like(close_12h)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(close_12h)):
        if basic_ub[i] < final_ub[i-1] or close_12h[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_12h[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend direction
    supertrend = np.zeros_like(close_12h)
    supertrend[0] = final_ub[0]
    for i in range(1, len(close_12h)):
        if supertrend[i-1] == final_ub[i-1]:
            if close_12h[i] <= final_ub[i]:
                supertrend[i] = final_ub[i]
            else:
                supertrend[i] = final_lb[i]
        else:
            if close_12h[i] >= final_lb[i]:
                supertrend[i] = final_lb[i]
            else:
                supertrend[i] = final_ub[i]
    
    # Trend direction: 1 for uptrend (price > Supertrend), -1 for downtrend (price < Supertrend)
    trend_dir = np.where(close_12h > supertrend, 1, -1)
    
    # 4h Donchian(20) bands
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h trend direction to 4h
    trend_dir_aligned = align_htf_to_ltf(prices, df_12h, trend_dir.astype(float))
    
    # Align 4h Donchian bands to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 34  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_dir_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h uptrend, price breaks above 4h Donchian upper band, volume spike
            if (trend_dir_aligned[i] == 1 and
                close[i] > donchian_high_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: 12h downtrend, price breaks below 4h Donchian lower band, volume spike
            elif (trend_dir_aligned[i] == -1 and
                  close[i] < donchian_low_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: trend reversal or price breaks below Donchian lower band
            if (trend_dir_aligned[i] == -1 or 
                close[i] < donchian_low_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or price breaks above Donchian upper band
            if (trend_dir_aligned[i] == 1 or 
                close[i] > donchian_high_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals