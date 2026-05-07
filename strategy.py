#!/usr/bin/env python3
"""
4h_Fractal_Pullback_Trend_Filter
Hypothesis: In trending markets, price often pulls back to the 50-period EMA before continuing.
Williams fractals identify swing points; we enter on pullbacks to EMA50 after a fractal forms,
with volume confirmation and ADX trend filter. Works in both bull (buy pullbacks) and bear (sell rallies).
Target: 20-40 trades per year (~80-160 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

name = "4h_Fractal_Pullback_Trend_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for fractals and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Williams fractals need 2-bar confirmation after the center bar
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Additional 2-bar delay for fractal confirmation (total 3 bars after center)
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    # 1-day EMA50 for trend and pullback reference
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ADX(14) on 1-day for trend strength filter
    # Calculate +DM, -DM, TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    )
    # Pad to same length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smoothed values
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume ratio: current volume / 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_14_aligned[i] > 25
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        # Price relative to EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0 and trending and volume_confirm:
            # Long: bullish fractal formed AND price pulling back to EMA50 from above
            # Bullish fractal = low point, wait for pullback to EMA
            long_setup = bullish_fractal_aligned[i] and price_above_ema and (close[i] <= ema_50_1d_aligned[i] * 1.02)
            # Short: bearish fractal formed AND price pulling back to EMA50 from below
            short_setup = bearish_fractal_aligned[i] and price_below_ema and (close[i] >= ema_50_1d_aligned[i] * 0.98)
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below EMA50 or trend weakens
            if (close[i] < ema_50_1d_aligned[i]) or (adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above EMA50 or trend weakens
            if (close[i] > ema_50_1d_aligned[i]) or (adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals