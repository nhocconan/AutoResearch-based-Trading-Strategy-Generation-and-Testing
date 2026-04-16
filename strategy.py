#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d ATR-based volatility breakout with 1w trend filter.
# Long when price breaks above 1d Donchian(20) high + ATR(14) expansion + 1w EMA50 uptrend.
# Short when price breaks below 1d Donchian(20) low + ATR(14) expansion + 1w EMA50 downtrend.
# Uses discrete position size 0.25. ATR filter ensures breakouts occur during expanding volatility.
# 1w EMA50 provides major trend filter to avoid counter-trend trades in choppy markets.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag on 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data once before loop for Donchian and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Donchian Channels (20-period) ===
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: ATR (14-period) for volatility expansion ===
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR expansion: current ATR > 1.2x ATR 10 periods ago
    atr_expansion = np.zeros_like(atr, dtype=bool)
    atr_expansion[10:] = atr[10:] > (atr[:-10] * 1.2)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (12h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    atr_expansion_aligned = align_htf_to_ltf(prices, df_1d, atr_expansion)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14, 10, 50)  # Donchian20, ATR14, ATR10lookback, EMA50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(atr_expansion_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        atr_exp = atr_expansion_aligned[i]
        ema50 = ema50_1w_aligned[i]
        price = close[i]
        
        # === EXIT LOGIC: Flip position on opposite signal ===
        exit_long = False
        exit_short = False
        
        # Exit long if short signal triggers
        if price < lower and atr_exp:
            exit_long = True
        
        # Exit short if long signal triggers
        if price > upper and atr_exp:
            exit_short = True
        
        if exit_long and position == 1:
            signals[i] = 0.0
            position = 0
            continue
            
        if exit_short and position == -1:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian high + ATR expansion + 1w uptrend
            if price > upper and atr_exp and price > ema50:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price breaks below Donchian low + ATR expansion + 1w downtrend
            elif price < lower and atr_exp and price < ema50:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dDonchian20_ATRExpansion_1wEMA50_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0