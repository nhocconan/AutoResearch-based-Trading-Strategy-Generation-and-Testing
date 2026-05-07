#!/usr/bin/env python3
name = "6h_Keltner_Channel_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Load daily data ONCE for Keltner channel and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Keltner Channel parameters (from daily)
    c_high = df_1d['high'].values
    c_low = df_1d['low'].values
    c_close = df_1d['close'].values
    
    # Calculate ATR(10) on daily
    tr1 = c_high[1:] - c_low[1:]
    tr2 = np.abs(c_high[1:] - c_close[:-1])
    tr3 = np.abs(c_low[1:] - c_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # EMA(20) as middle line
    ema_20 = pd.Series(c_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner upper/lower bands (multiplier = 1.5)
    kc_upper = ema_20 + (atr_10 * 1.5)
    kc_lower = ema_20 - (atr_10 * 1.5)
    
    # Align Keltner channels to 6h timeframe
    kc_upper_6h = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_6h = align_htf_to_ltf(prices, df_1d, kc_lower)
    ema_20_6h = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(c_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(kc_upper_6h[i]) or np.isnan(kc_lower_6h[i]) or 
            np.isnan(ema_50_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above upper Keltner in daily uptrend with volume
            if close[i] > kc_upper_6h[i] and ema_50_6h[i] > ema_50_6h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Keltner in daily downtrend with volume
            elif close[i] < kc_lower_6h[i] and ema_50_6h[i] < ema_50_6h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to EMA20 or trend reverses
            if close[i] < ema_20_6h[i] or ema_50_6h[i] < ema_50_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to EMA20 or trend reverses
            if close[i] > ema_20_6h[i] or ema_50_6h[i] > ema_50_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Keltner Channel breakouts with daily trend filter and volume confirmation
# - Keltner Channel (EMA20 ± 1.5*ATR10) adapts to volatility, providing dynamic support/resistance
# - Breakout above upper band in daily uptrend (EMA50 rising) signals bullish continuation
# - Breakdown below lower band in daily downtrend (EMA50 falling) signals bearish continuation
# - Volume confirmation (1.5x average) reduces false breakouts
# - Exit when price returns to EMA20 (middle line) or daily trend reverses
# - Position size 0.25 targets ~20-40 trades/year to avoid fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses 1d timeframe for structure and trend, 6h for execution timing
# - Novelty: Adaptive channels + trend filter + volume combo not recently tried on 6h
# - Expected to perform well in ranging markets (channel bounds) and trending markets (breakouts)