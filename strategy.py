#!/usr/bin/env python3
# 4h_1d_keltner_breakout_volume_v1
# Strategy: 4h Keltner Channel breakout with volume confirmation and 1D ADX trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Keltner Channel breakouts capture volatility expansion in trending markets.
# Volume confirmation ensures institutional participation. 1D ADX > 25 filters for trending regimes.
# Works in bull/bear by following breakout direction. Target: 20-40 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1D data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1D ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx = np.concatenate([[np.nan] * 13, adx[13:]])  # pad for warmup
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 4H Keltner Channel (20, 2.0)
    ema_middle = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_4h = pd.Series(np.maximum(high - low, 
                                  np.maximum(np.abs(high - np.roll(close, 1)), 
                                             np.abs(low - np.roll(close, 1))))).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_channel = ema_middle + 2.0 * atr_4h
    lower_channel = ema_middle - 2.0 * atr_4h
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_middle[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Entry logic: Channel breakout + volume + trend
        if (close[i] > upper_channel[i] and vol_confirm[i] and trending and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < lower_channel[i] and vol_confirm[i] and trending and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Channel reversion or loss of trend
        elif position == 1 and (close[i] < ema_middle[i] or not trending):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_middle[i] or not trending):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals