#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX trend filter
# Works in bull (breakouts continue) and bear (false breakouts filtered by ADX < 25)
# Target: 20-40 trades/year per symbol to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian(20) and ADX - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from daily data
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    upper_20 = pd.Series(high_daily).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_daily).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX(14) from daily data for trend strength
    # TR = max(high-low, abs(high-previous_close), abs(low-previous_close))
    prev_close = np.concatenate([[high_daily[0]], high_daily[:-1]])
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - prev_close)
    tr3 = np.abs(low_daily - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional movement
    up_move = high_daily - np.concatenate([[high_daily[0]], high_daily[:-1]])
    down_move = np.concatenate([[low_daily[0]], low_daily[:-1]]) - low_daily
    up_move = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_move = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    up14 = pd.Series(up_move).rolling(window=14, min_periods=14).sum().values
    down14 = pd.Series(down_move).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    plus_di = 100 * up14 / tr14
    minus_di = 100 * down14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_daily, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_daily, lower_20)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume average (20-period) on 4h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian(20) with volume AND strong trend (ADX > 25)
            if (close[i] > upper_20_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i] and
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian(20) with volume AND strong trend (ADX > 25)
            elif (close[i] < lower_20_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i] and
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Donchian channel
            if position == 1:
                if close[i] < lower_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_Volume_ADX"
timeframe = "4h"
leverage = 1.0