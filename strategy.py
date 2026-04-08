#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Fractal Breakout with 12h Trend and Volume Confirmation
# Uses Williams Fractal to identify swing points, then breaks above/below
# recent fractal highs/lows in the direction of the 12h trend (ADX > 25).
# Volume confirmation (>1.5x 20-period average) filters weak breakouts.
# Works in bull/bear by trading breakouts in trend direction.
# Target: 20-50 trades/year via strict fractal + trend + volume confluence.
name = "4h_fractal_breakout_12h_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 14-period ADX for trend strength on 12h data
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
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
    
    # Williams Fractal (5-bar: need 2 bars on each side)
    # Bullish fractal: low[n-2] < low[n-1] and low[n] < low[n-1] and low[n+1] > low[n] and low[n+2] > low[n]
    # Bearish fractal: high[n-2] > high[n-1] and high[n] > high[n-1] and high[n+1] < high[n] and high[n+2] < high[n]
    fractal_high = np.zeros(n, dtype=bool)
    fractal_low = np.zeros(n, dtype=bool)
    
    for i in range(2, n-2):
        if (high[i-2] > high[i-1] and high[i] > high[i-1] and 
            high[i+1] < high[i] and high[i+2] < high[i]):
            fractal_high[i] = True
        if (low[i-2] < low[i-1] and low[i] < low[i-1] and 
            low[i+1] > low[i] and low[i+2] > low[i]):
            fractal_low[i] = True
    
    # Most recent fractal high/low (updated when new fractal forms)
    last_fractal_high = np.full(n, np.nan)
    last_fractal_low = np.full(n, np.nan)
    
    last_high_val = np.nan
    last_low_val = np.nan
    for i in range(n):
        if fractal_high[i]:
            last_high_val = high[i]
        if fractal_low[i]:
            last_low_val = low[i]
        last_fractal_high[i] = last_high_val
        last_fractal_low[i] = last_low_val
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(last_fractal_high[i]) or 
            np.isnan(last_fractal_low[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 12h values for current 4h bar
        adx_aligned = align_htf_to_ltf(prices, df_12h, adx)[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned > 25
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below last fractal low OR trend weakens
            if close[i] < last_fractal_low[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above last fractal high OR trend weakens
            if close[i] > last_fractal_high[i] or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Only trade during strong trend with volume confirmation
            if strong_trend and volume_confirm:
                # Long: price breaks above last fractal high
                if close[i] > last_fractal_high[i]:
                    position = 1
                    signals[i] = 0.30
                # Short: price breaks below last fractal low
                elif close[i] < last_fractal_low[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals