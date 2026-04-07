#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and 1w volatility regime
# Uses Donchian channel breakouts for entries, 1d EMA(50) for trend filter,
# and 1w ATR percentile to scale position size based on volatility regime.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
# Works in bull markets via breakout momentum and in bear via mean reversion at extremes.

name = "12h_donchian20_1d_ema_1w_vol_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1w ATR(14) for volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1w_ma = pd.Series(atr_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_1w_ma_aligned = align_htf_to_ltf(prices, df_1w, atr_1w_ma)
    
    # Donchian(20) on 12h data
    highest = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(atr_1w_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime: scale position based on ATR percentile
        vol_ratio = atr_1w_aligned[i] / atr_1w_ma_aligned[i] if atr_1w_ma_aligned[i] > 0 else 1.0
        # In low vol (trending): increase size, in high vol (choppy): decrease
        if vol_ratio < 0.8:
            vol_scale = 1.3  # Increase size in low vol
        elif vol_ratio > 1.2:
            vol_scale = 0.7  # Decrease size in high vol
        else:
            vol_scale = 1.0  # Normal size
        
        base_size = 0.25
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest[i-1]  # Break above 20-period high
        breakout_down = close[i] < lowest[i-1]  # Break below 20-period low
        
        # 1d EMA trend filter
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Long: bullish breakout in uptrend OR bearish breakout in downtrend (mean reversion)
        if breakout_up and uptrend:
            signals[i] = base_size * vol_scale
        # Short: bearish breakout in downtrend OR bullish breakout in uptrend (mean reversion)
        elif breakout_down and downtrend:
            signals[i] = -base_size * vol_scale
        else:
            signals[i] = 0.0
    
    return signals