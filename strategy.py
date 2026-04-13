#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d volume confirmation and chop regime filter
    # Uses Donchian(20) for structure, 1d volume for confirmation, and choppiness index for regime
    # Discrete sizing (0.25) to minimize fee drag
    # Target: 15-35 trades/year to stay within 4h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for volume and chop regime (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d ATR(14) for choppiness index
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align length
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d true range sum and ATR sum for choppiness
    tr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    atr_sum_14 = atr_1d * 14  # approximate
    chop_1d = 100 * np.log10(tr_sum_14 / atr_sum_14) / np.log10(14)
    
    # Align 1d indicators to 4h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    entry_price = np.full(n, np.nan)
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop > 61.8 = ranging (mean reversion), chop < 38.2 = trending
        # In ranging markets, we fade Donchian touches; in trending, we breakout
        is_ranging = chop_1d_aligned[i] > 61.8
        is_trending = chop_1d_aligned[i] < 38.2
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        idx_1d = i // 96  # 4h bars per 1d (24*60/4/4 = 96)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Donchian breakout/fade logic
        breakout_long = (close[i] > highest_high[i]) and volume_confirmed and is_trending
        breakout_short = (close[i] < lowest_low[i]) and volume_confirmed and is_trending
        fade_long = (close[i] < lowest_low[i]) and volume_confirmed and is_ranging
        fade_short = (close[i] > highest_high[i]) and volume_confirmed and is_ranging
        
        # ATR-based stoploss (using 1d ATR)
        idx_1d_atr = i // 96
        if idx_1d_atr < len(atr_1d) and not np.isnan(atr_1d[idx_1d_atr]):
            stop_distance = atr_1d[idx_1d_atr] * 2.0  # 2x ATR stop
        else:
            stop_distance = 0
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif fade_long and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif fade_short and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "4h_1d_donchian_chop_volume_v1"
timeframe = "4h"
leverage = 1.0