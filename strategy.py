#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h strategy using 1d Donchian channel breakouts with volume confirmation and chop regime filter
    # Works in bull/bear: Donchian breakouts capture strong moves, volume confirms institutional participation,
    # chop filter avoids whipsaws in ranging markets. Discrete sizing (0.25) minimizes fee drag.
    # Target: 20-40 trades/year to stay within 4h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Get 1d data for choppiness index calculation
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d True Range for choppiness index
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ADX for trend strength (simplified chop filter)
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    
    atr_14_for_dx = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14_for_dx
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14_for_dx
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    dx = np.where((plus_di_14 + minus_di_14) == 0, 0, dx)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume average for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h primary timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high_20_aligned[i]) or 
            np.isnan(lowest_low_20_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        idx_1d = i // (24 * 6)  # 1d bars in 4h timeframe (6 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: ADX > 20 indicates trending market (avoid chop)
        trending_market = adx_14_aligned[i] > 20
        
        # Entry conditions: Donchian breakout + volume + trend filter
        enter_long = (close[i] > highest_high_20_aligned[i]) and volume_confirmed and trending_market
        enter_short = (close[i] < lowest_low_20_aligned[i]) and volume_confirmed and trending_market
        
        # Stoploss: 1.5x ATR based on 1d true range
        if idx_1d < len(true_range):
            atr_value = true_range[idx_1d]
            stop_distance = atr_value * 1.5
        else:
            stop_distance = 0
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
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

name = "4h_1d_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0