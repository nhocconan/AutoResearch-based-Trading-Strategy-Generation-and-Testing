#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with 1-week EMA trend filter and volume confirmation
# Works in bull markets via breakouts, works in bear via shorting breakdowns
# Uses 1d timeframe targeting 15-30 trades/year to minimize fee drag
# Volume confirmation ensures breakout validity, 1w EMA provides major trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    highest_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate 1w EMA(34) for major trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d ATR(14) for volume confirmation and position sizing
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d average volume (20-period) for confirmation
    avg_vol_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(avg_vol_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * avg_vol_20_aligned[i]
        
        # Long conditions:
        # 1. Price breaks above 20-day high (Donchian breakout)
        # 2. Price above 1-week EMA34 (bullish major trend)
        # 3. Volume confirmation
        if (close[i] > highest_20_aligned[i] and
            close[i] > ema_34_1w_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 20-day low (Donchian breakdown)
        # 2. Price below 1-week EMA34 (bearish major trend)
        # 3. Volume confirmation
        elif (close[i] < lowest_20_aligned[i] and
              close[i] < ema_34_1w_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_EMA34w_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0