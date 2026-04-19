#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + 1d ADX trend filter
# Entry: Price breaks above Donchian upper (long) or below Donchian lower (short)
#        Confirmed by 1d volume > 1.5x 20-period average and 1d ADX > 25
# Exit: Opposite Donchian break or 2x ATR stop
# Designed to work in trending markets (both bull and bear) by capturing breakouts
# Target: 15-25 trades/year to minimize fee drag
name = "12h_Donchian20_1dVolADX_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and ADX filters (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 1d ADX calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_period = 14
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/tr_period, adjust=False).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/tr_period, adjust=False).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/tr_period, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 12h Donchian channels (20-period)
    donch_len = 20
    donch_high = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    donch_low = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # 12h ATR for stops
    tr_12h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_12h[0] = high[0] - low[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(vol_avg_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or \
           np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr_12h[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x 20-period average
        vol_filter = vol_1d[i] > 1.5 * vol_avg_1d[i] if not np.isnan(vol_avg_1d[i]) else False
        
        # ADX filter: trend strength
        adx_filter = adx_1d[i] > 25
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume + ADX
            if close[i] > donch_high[i-1] and vol_filter and adx_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume + ADX
            elif close[i] < donch_low[i-1] and vol_filter and adx_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below Donchian low or 2x ATR stop
            if close[i] < donch_low[i-1] or close[i] < close[i-1] - 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above Donchian high or 2x ATR stop
            if close[i] > donch_high[i-1] or close[i] > close[i-1] + 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals