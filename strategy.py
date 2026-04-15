#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20) with 1d ADX trend filter and volume confirmation
# Long when price breaks above 20-period high + ADX>25 + volume>1.5x median
# Short when price breaks below 20-period low + ADX>25 + volume>1.5x median
# Exit when price crosses back inside the channel or ADX<20 (trend weakening)
# Works in bull (breakouts continue) and bear (strong trends persist)
# Uses discrete sizing (0.25) to limit overtrading and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4-hour Donchian Channel (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(adx_aligned[i])):
            continue
        
        # Long: breakout above upper band + strong trend + volume
        if (close[i] > donch_high[i] and 
            adx_aligned[i] > 25 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: breakout below lower band + strong trend + volume
        elif (close[i] < donch_low[i] and 
              adx_aligned[i] > 25 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back inside channel OR trend weakens (ADX<20)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donch_high[i]) or
               (signals[i-1] == -0.25 and close[i] > donch_low[i]) or
               adx_aligned[i] < 20)):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Breakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0