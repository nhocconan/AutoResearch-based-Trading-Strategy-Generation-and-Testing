#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Trend_1d_Volume_Confirm_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 120:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and regime (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA34 for trend filter (trend confirmed after daily close)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily ATR for regime filter (trending vs ranging)
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr10_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # 4h Donchian channel (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h ATR for exit conditions
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr10_4h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 120
    
    for i in range(start_idx, n):
        if np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(vol_ma_20[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(atr10_1d_aligned[i]) or np.isnan(atr10_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr10_4h[i]
        ema_trend = ema34_1d_aligned[i]
        atr1d = atr10_1d_aligned[i]
        
        # Volume confirmation: need strong volume breakout
        volume_confirmed = vol > 2.0 * vol_ma
        
        # Regime filter: only trade in trending markets (daily ATR > 1.5x 50-period average)
        if i >= 50:
            atr1d_ma_50 = np.nanmean(atr10_1d_aligned[i-50:i]) if not np.any(np.isnan(atr10_1d_aligned[i-50:i])) else np.nan
            trending = atr1d > 1.5 * atr1d_ma_50 if not np.isnan(atr1d_ma_50) else True
        else:
            trending = True  # insufficient data, allow trade
        
        if position == 0:
            # Long: price breaks above upper Donchian + uptrend + volume + trending regime
            if price > high_max_20[i] and price > ema_trend and volume_confirmed and trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + downtrend + volume + trending regime
            elif price < low_min_20[i] and price < ema_trend and volume_confirmed and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below EMA34 or ATR trailing stop (2.5x ATR)
            if price < ema_trend or price < (high[i] - 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above EMA34 or ATR trailing stop (2.5x ATR)
            if price > ema_trend or price > (low[i] + 2.5 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals