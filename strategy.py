#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get daily data for volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # Daily ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1h ATR for entry trigger
    tr1h = np.abs(high - low)
    tr2h = np.abs(high - np.roll(close, 1))
    tr3h = np.abs(low - np.roll(close, 1))
    trh = np.maximum(tr1h, np.maximum(tr2h, tr3h))
    trh[0] = tr1h[0]
    atr_1h = pd.Series(trh).rolling(window=14, min_periods=14).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_4h_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(atr_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA200
        uptrend = close[i] > ema200_4h_aligned[i]
        downtrend = close[i] < ema200_4h_aligned[i]
        
        # Volatility regime: only trade when volatility is elevated
        vol_regime = atr_1h[i] > 1.5 * atr_1d_aligned[i]
        
        # Entry triggers: price breaks 4h EMA200 with volatility
        long_entry = (close[i] > ema200_4h_aligned[i]) and vol_regime
        short_entry = (close[i] < ema200_4h_aligned[i]) and vol_regime
        
        # Exit: trend reversal or volatility contraction
        long_exit = (close[i] < ema200_4h_aligned[i]) or (atr_1h[i] < 0.8 * atr_1d_aligned[i])
        short_exit = (close[i] > ema200_4h_aligned[i]) or (atr_1h[i] < 0.8 * atr_1d_aligned[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_EMA200_Trend_VolatilityRegime_Session"
timeframe = "1h"
leverage = 1.0