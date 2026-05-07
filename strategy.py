#!/usr/bin/env python3
name = "12h_Keltner_Breakout_TrendVolume_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Load weekly data ONCE before loop for ATR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR(14) on weekly
    tr1 = np.maximum(df_1w['high'].values, np.roll(df_1w['close'].values, 1))
    tr1 = np.maximum(tr1, np.roll(df_1w['low'].values, 1))
    tr2 = np.abs(df_1w['high'].values - np.roll(df_1w['low'].values, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = tr[1]  # first bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Calculate EMA(20) on daily
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Keltner Channels on 12h: EMA(20) ± 2*ATR
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper = ema_20 + 2 * atr_14_aligned
    lower = ema_20 - 2 * atr_14_aligned
    
    # Volume spike: 4-bar average (2 days of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 4)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above upper Keltner with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_20_1d_aligned[i] > ema_20_1d_aligned[i-1]
            
            if close[i] > upper[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: close below lower Keltner with volume and daily downtrend
            elif close[i] < lower[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close back below EMA(20) or volume drops
            if close[i] < ema_20[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close back above EMA(20) or volume drops
            if close[i] > ema_20[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Keltner Channel breakout with weekly ATR and daily trend
# - Keltner Channels (EMA ± 2*ATR) adapt to volatility better than fixed bands
# - Weekly ATR(14) provides stable volatility measure less prone to whipsaw
# - Breakout above upper band with volume in daily uptrend = long
# - Breakdown below lower band with volume in daily downtrend = short
# - Volume spike (2x average) confirms institutional participation
# - Exit when price returns to EMA(20) or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Weekly ATR ensures volatility normalization across regimes
# - Daily trend filter prevents counter-trend entries in choppy markets
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)