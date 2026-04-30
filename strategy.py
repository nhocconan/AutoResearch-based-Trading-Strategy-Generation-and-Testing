#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation + ATR trailing stop.
# Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 2.0x 20-bar average.
# Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 2.0x 20-bar average.
# Uses discrete position sizing (0.25) to limit drawdown and fee churn. Works in bull/bear via 1d EMA34 trend filter.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # ATR for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, lookback)  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        is_uptrend = close[i] > ema_34_1d_aligned[i]
        is_downtrend = close[i] < ema_34_1d_aligned[i]
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high + uptrend + volume confirmation
            if curr_high > highest_high[i-1] and is_uptrend and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: break below Donchian low + downtrend + volume confirmation
            elif curr_low < lowest_low[i-1] and is_downtrend and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_since_entry:
                highest_since_entry = curr_high
            
            # Trailing stop: 2.5 * ATR below highest since entry
            if curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_since_entry:
                lowest_since_entry = curr_low
            
            # Trailing stop: 2.5 * ATR above lowest since entry
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals