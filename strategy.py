#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_Volume_Trend_1d1w_v1"
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
    
    # Get 1d and 1w data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d EMA34 and 1w EMA34 for trend
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Donchian channel (20-period) on 12h
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for exit conditions
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(atr_10[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_10[i]
        ema_trend_1d = ema34_1d_aligned[i]
        ema_trend_1w = ema34_1w_aligned[i]
        
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper Donchian + uptrend on both 1d and 1w + volume
            if price > high_max_20[i] and price > ema_trend_1d and price > ema_trend_1w and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + downtrend on both 1d and 1w + volume
            elif price < low_min_20[i] and price < ema_trend_1d and price < ema_trend_1w and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below EMA34 on either 1d or 1w or ATR trailing stop
            if price < ema_trend_1d or price < ema_trend_1w or price < (high[i] - 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above EMA34 on either 1d or 1w or ATR trailing stop
            if price > ema_trend_1d or price > ema_trend_1w or price > (low[i] + 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals