#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND 1d ADX > 25 AND volume > 1.5x 24-period average
# Short when price breaks below Donchian lower band AND 1d ADX > 25 AND volume > 1.5x 24-period average
# Uses ATR-based trailing stop (2.0x ATR) for risk management
# Discrete position sizing (0.25) to balance return and fee drag
# Target: 20-40 trades/year on 12h timeframe to avoid fee drag while capturing strong breakouts
# Works in trending markets via ADX filter (avoids choppy periods)
# Volume confirmation ensures breakouts have strong participation
# ADX > 25 ensures we only trade when there is a strong trend

name = "12h_Donchian_Breakout_1dADX25_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend filter (period=14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_first = np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])
    tr_1d = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing, equivalent to EMA with alpha=1/period)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # Directional Index (DX) and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate ATR for stoploss (using 24-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    # Calculate Donchian channels (20-period) on 12h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, lookback)  # warmup for Donchian and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_adx_1d = adx_1d_aligned[i]
        curr_atr = atr[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        # Volume spike confirmation: current volume > 1.5x 24-period average
        if i >= 24:
            vol_ma_24 = np.mean(volume[i-24:i])
        else:
            vol_ma_24 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_24 if vol_ma_24 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR Donchian break below lower band
            if curr_close < stop_price or curr_close < curr_lowest_low:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop OR Donchian break above upper band
            if curr_close > stop_price or curr_close > curr_highest_high:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band AND 1d ADX > 25 AND volume spike
            if curr_close > curr_highest_high and curr_adx_1d > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below Donchian lower band AND 1d ADX > 25 AND volume spike
            elif curr_close < curr_lowest_low and curr_adx_1d > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals