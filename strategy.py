#!/usr/bin/env python3
# 12h_fractal_breakout_1d_trend_volume_v3
# Hypothesis: 12h timeframe trading using daily Williams Fractal breakouts with volume confirmation and ADX trend filter. Fractals provide key support/resistance levels; breakouts with volume capture momentum in both bull and bear markets. Daily trend filter (ADX) ensures alignment with higher timeframe direction. Target: 20-40 trades/year per symbol.

name = "12h_fractal_breakout_1d_trend_volume_v3"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (up) and bullish (down) fractals."""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest of 5 bars
        if high[i] >= high[i-1] and high[i] >= high[i-2] and high[i] >= high[i+1] and high[i] >= high[i+2]:
            bearish[i] = high[i]
        # Bullish fractal: low[i] is lowest of 5 bars
        if low[i] <= low[i-1] and low[i] <= low[i-2] and low[i] <= low[i+1] and low[i] <= low[i+2]:
            bullish[i] = low[i]
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for fractals and trend filter - call ONCE before loop
    df_d = get_htf_data(prices, '1d')
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Calculate daily ADX for trend strength filter
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    up_move = high_d - np.roll(high_d, 1)
    down_move = np.roll(low_d, 1) - low_d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx = np.where((plus_di + minus_di) == 0, 0, adx)
    adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
    
    # Calculate daily Williams Fractals
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_d, low_d)
    # Need 2-bar confirmation for fractals (wait for 2 candles after the fractal)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_d, bullish_fractal, additional_delay_bars=2)
    
    # Calculate 20-period average volume for 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned daily indicators for current 12h bar
        adx_val = align_htf_to_ltf(prices, df_d, adx)[i]
        bearish_val = bearish_fractal_aligned[i]
        bullish_val = bullish_fractal_aligned[i]
        
        # Skip if any required data is NaN
        if np.isnan(adx_val) or np.isnan(vol_ma[i]) or volume[i] == 0:
            signals[i] = 0.0
            continue
        
        # Volume breakout condition: current volume > 1.5x 20-period average
        vol_breakout = volume[i] > 1.5 * vol_ma[i]
        
        # Strong trend condition: ADX > 25
        strong_trend = adx_val > 25
        
        if position == 1:  # Long position
            # Exit if price breaks below bullish fractal (support)
            if not np.isnan(bullish_val) and close[i] < bullish_val:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above bearish fractal (resistance)
            if not np.isnan(bearish_val) and close[i] > bearish_val:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above bearish fractal (resistance) with volume confirmation and strong trend
            if not np.isnan(bearish_val) and high[i] >= bearish_val and close[i] > bearish_val and vol_breakout and strong_trend:
                position = 1
                signals[i] = 0.25
            # Breakout short below bullish fractal (support) with volume confirmation and strong trend
            elif not np.isnan(bullish_val) and low[i] <= bullish_val and close[i] < bullish_val and vol_breakout and strong_trend:
                position = -1
                signals[i] = -0.25
    
    return signals