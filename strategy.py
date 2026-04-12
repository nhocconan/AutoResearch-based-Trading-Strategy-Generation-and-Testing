#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend direction (1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Donchian channels (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Daily ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        atr_1d[i] = np.nanmean(tr[i-14:i+1])
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume filter: 20-period EMA on 6h volume
    vol_ema = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema[:] = vol_ema_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ema[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x EMA
        volume_filter = volume[i] > vol_ema[i] * 1.5
        
        # Volatility filter: ATR > 0.5 * 20-period ATR mean
        atr_ma = np.full(n, np.nan)
        if i >= 34:
            atr_ma[i] = np.nanmean(atr_1d_aligned[i-20:i])
        vol_filter = atr_1d_aligned[i] > atr_ma[i] * 0.5 if not np.isnan(atr_ma[i]) else True
        
        # Trend filter: price above/below weekly EMA200
        uptrend = close[i] > ema200_1w_aligned[i]
        downtrend = close[i] < ema200_1w_aligned[i]
        
        # Breakout conditions: Donchian breakout with trend, volume, and volatility
        long_breakout = (high[i] > donch_high_aligned[i]) and uptrend and volume_filter and vol_filter
        short_breakout = (low[i] < donch_low_aligned[i]) and downtrend and volume_filter and vol_filter
        
        # Exit conditions: Opposite Donchian band touch
        long_exit = low[i] < donch_low_aligned[i]
        short_exit = high[i] > donch_high_aligned[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_donchian_breakout_trend_vol_filter_v1"
timeframe = "6h"
leverage = 1.0