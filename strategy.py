#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Uses discrete sizing 0.30 to balance profit and fee drag. Target: 100-200 total trades over 4 years (25-50/year).
# Donchian(20) provides clear structure; 1d ADX > 25 filters for trending markets only.
# Volume spike ensures institutional participation. Works in both bull and bear via 1d trend filter.

name = "4h_Donchian20_1dADX25_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 1d ADX(14) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range calculation for ADX
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = df_1d['high'].diff()
    down_move = -df_1d['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_14.values
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_14.values
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_14_1d = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 2.0x 30-period average (strict to reduce trades)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma_30)
    
    # ATR for stoploss (14-period)
    tr1_l = high[1:] - low[1:]
    tr2_l = np.abs(high[1:] - close[:-1])
    tr3_l = np.abs(low[1:] - close[:-1])
    tr_l = np.concatenate([[np.nan], np.maximum(tr1_l, np.maximum(tr2_l, tr3_l))])
    atr_14_l = pd.Series(tr_l).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(lookback, 30, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(adx_14_1d[i]) or np.isnan(vol_ma_30[i]) or
            np.isnan(atr_14_l[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_adx = adx_14_1d[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14_l[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 1d ADX > 25 trend filter
            if curr_volume_spike and curr_adx > 25:
                # Bullish: Close breaks above upper Donchian
                if curr_close > curr_highest_high:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below lower Donchian
                elif curr_close < curr_lowest_low:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry
            stop_loss = entry_price - 2.0 * curr_atr
            # Exit: Stoploss hit OR close drops below lower Donchian OR ADX < 20 (trend weak)
            if curr_low <= stop_loss or curr_close < curr_lowest_low or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry
            stop_loss = entry_price + 2.0 * curr_atr
            # Exit: Stoploss hit OR close rises above upper Donchian OR ADX < 20 (trend weak)
            if curr_high >= stop_loss or curr_close > curr_highest_high or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals