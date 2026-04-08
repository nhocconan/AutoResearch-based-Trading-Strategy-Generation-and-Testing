#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour fractal breakout with 1-week trend filter (ADX>30) and volume confirmation (volume > 1.5x 10-period average)
# Fractal breakouts capture turning points, weekly ADX filters for strong trends, volume confirms breakout strength
# Works in both bull and bear by capturing breakouts in either direction
# Target: 12-37 trades/year by requiring fractal breakout + trend + volume confirmation
name = "12h_fractal_breakout_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # Get 1w data for ADX (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ADX for trend strength on 1w data
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 12h fractal indicator (5-bar pattern)
    # Bullish fractal: low < low[-2] and low < low[-1] and low < low[+1] and low < low[+2]
    # Bearish fractal: high > high[-2] and high > high[-1] and high > high[+1] and high > high[+2]
    bullish_fractal = np.zeros(n, dtype=bool)
    bearish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = True
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = True
    
    # 10-period volume average for confirmation
    vol_avg_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(vol_avg_10[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1w values for current 12h bar
        adx_aligned = align_htf_to_ltf(prices, df_1w, adx)[i]
        
        # Trend filter: ADX > 30 indicates strong trend
        strong_trend = adx_aligned > 30
        
        # Volume confirmation: current volume > 1.5x 10-period average
        volume_confirm = volume[i] > 1.5 * vol_avg_10[i]
        
        if position == 1:  # Long position
            # Exit: bearish fractal forms OR trend weakens
            if bearish_fractal[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish fractal forms OR trend weakens
            if bullish_fractal[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade during strong trend with volume confirmation
            if strong_trend and volume_confirm:
                # Long: bullish fractal forms
                if bullish_fractal[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: bearish fractal forms
                elif bearish_fractal[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals