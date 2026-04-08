#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_atr_breakout_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR on 1d (14-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Trend: 1d EMA20
    ema_1d = pd.Series(close_1d).ewm(span=20, min_periods=20).mean().values
    
    # Align to 4h timeframe
    atr_1d_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1d_4h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # ATR on 4h (14-period) for breakout threshold
    tr4h1 = high - low
    tr4h2 = np.abs(high - np.roll(close, 1))
    tr4h3 = np.abs(low - np.roll(close, 1))
    tr4h = np.maximum(tr4h1, np.maximum(tr4h2, tr4h3))
    tr4h[0] = tr4h1[0]
    atr_4h = pd.Series(tr4h).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channel (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_4h[i]) or np.isnan(ema_1d_4h[i]) or
            np.isnan(atr_4h[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: close below Donchian low or trend reversal
            if close[i] < donchian_low[i] or close[i] < ema_1d_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above Donchian high or trend reversal
            if close[i] > donchian_high[i] or close[i] > ema_1d_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish_trend = close[i] > ema_1d_4h[i]
            bearish_trend = close[i] < ema_1d_4h[i]
            
            # Breakout filter
            bullish_breakout = close[i] > donchian_high[i]
            bearish_breakout = close[i] < donchian_low[i]
            
            # Long: bullish trend + bullish breakout + volume
            if (bullish_trend and bullish_breakout and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: bearish trend + bearish breakout + volume
            elif (bearish_trend and bearish_breakout and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals