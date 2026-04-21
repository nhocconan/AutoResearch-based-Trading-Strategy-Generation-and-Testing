#!/usr/bin/env python3
"""
4h_Bollinger_Band_Breakout_Volume_TrendFilter
Hypothesis: Breakouts beyond Bollinger Bands (20,2) with volume confirmation and aligned daily trend (EMA50) yield high-probability trades. Uses Bollinger Band squeeze as a volatility filter to avoid chop. Works in bull/bear markets by only taking breakouts in direction of daily trend. Target: 20-50 trades/year on 4h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands on 4h close
    close = prices['close'].values
    bb_length = 20
    bb_mult = 2.0
    basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_band = basis + dev
    lower_band = basis - dev
    
    # Bollinger Band width for squeeze detection (avoid chop)
    bb_width = (upper_band - lower_band) / basis
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma  # True when volatility is low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if EMA not ready
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price > EMA50 for long, price < EMA50 for short
        trend_long = price > ema_50_1d_aligned[i]
        trend_short = price < ema_50_1d_aligned[i]
        
        # Volatility filter: only trade when not in squeeze (avoid chop)
        vol_filter = not bb_squeeze[i] if i < len(bb_squeeze) else True
        
        if position == 0:
            # Long: price breaks above upper BB + volume confirmation + uptrend + volatility filter
            if price > upper_band[i] and volume_ok and trend_long and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB + volume confirmation + downtrend + volatility filter
            elif price < lower_band[i] and volume_ok and trend_short and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band or trend turns bearish
            if price < basis[i] or not trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band or trend turns bullish
            if price > basis[i] or not trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Bollinger_Band_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0