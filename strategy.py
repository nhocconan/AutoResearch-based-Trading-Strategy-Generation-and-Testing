#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume spike and 1w ADX trend filter
# Uses 12h timeframe to target 12-37 trades/year. Breakouts filtered by 1d volume > 2x median and 1w ADX > 25.
# Works in bull (breakouts continue) and bear (avoids false breakouts in low ADX).
# Discrete position sizing (0.25) limits fee churn.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day volume for spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_median_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median()
    vol_spike = volume_1d > (2.0 * vol_median_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # 1-week ADX for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])
    down_move = np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr_1w).ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_ma
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_ma
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_filter = adx_values > 25
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_filter)
    
    # 12-hour Donchian channels (20 periods)
    donch_window = 20
    highest_high = pd.Series(high).rolling(window=donch_window, min_periods=donch_window).max()
    lowest_low = pd.Series(low).rolling(window=donch_window, min_periods=donch_window).min()
    
    signals = np.zeros(n)
    
    for i in range(donch_window, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long: close breaks above Donchian high + volume spike + ADX filter
        if close[i] > highest_high[i] and vol_spike_aligned[i] and adx_aligned[i]:
            signals[i] = 0.25
        
        # Short: close breaks below Donchian low + volume spike + ADX filter
        elif close[i] < lowest_low[i] and vol_spike_aligned[i] and adx_aligned[i]:
            signals[i] = -0.25
        
        # Exit: close crosses back inside Donchian channels
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < highest_high[i]) or
               (signals[i-1] == -0.25 and close[i] > lowest_low[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0