#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA 50 on daily for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Weekly timeframe for regime filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # RSI 14 on weekly for overbought/oversold
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    avg_gain_1w = pd.Series(gain_1w).rolling(window=14, min_periods=14).mean().values
    avg_loss_1w = pd.Series(loss_1w).rolling(window=14, min_periods=14).mean().values
    rs_1w = avg_gain_1w / np.where(avg_loss_1w == 0, np.nan, avg_loss_1w)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w = np.where(np.isnan(rs_1w), 50, rsi_1w)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Donchian channels (20-period) on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or \
           np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR weekly RSI overbought
            if close[i] <= donchian_low[i] or rsi_1w_aligned[i] > 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR weekly RSI oversold
            if close[i] >= donchian_high[i] or rsi_1w_aligned[i] < 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume and weekly RSI not extreme
            if not volume_ok[i] or rsi_1w_aligned[i] > 70 or rsi_1w_aligned[i] < 30:
                signals[i] = 0.0
                continue
            
            # Long: price breaks above Donchian high AND daily close above EMA50
            if close[i] > donchian_high[i] and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND daily close below EMA50
            elif close[i] < donchian_low[i] and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals