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
    
    # Load 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h ATR(14) for volatility filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_init = np.array([max(high_12h[0] - low_12h[0], abs(high_12h[0] - close_12h[0]), abs(low_12h[0] - close_12h[0]))])
    tr = np.concatenate([tr_init, np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    # 12h EMA(50) for trend
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # 4h Donchian(20) breakout
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Get aligned 12h indicators
        atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)[i]
        ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)[i]
        
        # Skip if not enough data
        if np.isnan(atr_aligned) or np.isnan(ema50_aligned) or np.isnan(high_4h[i]) or np.isnan(low_4h[i]):
            continue
        
        # Volatility filter: only trade when volatility is above average
        vol_filter = atr_aligned > np.nanmedian(atr_12h)
        
        # Trend filter: price above/below 12h EMA50
        if close[i] > ema50_aligned:
            # Bullish trend: long on Donchian breakout
            if close[i] > high_4h[i] and vol_filter and position <= 0:
                position = 1
                signals[i] = position_size
        else:
            # Bearish trend: short on Donchian breakdown
            if close[i] < low_4h[i] and vol_filter and position >= 0:
                position = -1
                signals[i] = -position_size
        
        # Exit on opposite Donchian signal or volatility collapse
        if position == 1 and (close[i] < low_4h[i] or atr_aligned < 0.5 * np.nanmedian(atr_12h)):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > high_4h[i] or atr_aligned < 0.5 * np.nanmedian(atr_12h)):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_12h_Donchian_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0