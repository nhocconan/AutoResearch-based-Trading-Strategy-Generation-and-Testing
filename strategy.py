#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1-week Williams Fractal breakout with volume confirmation and ADX filter.
Weekly fractals provide strong support/resistance levels. Breakouts above/below these levels with
volume surge and ADX > 25 indicate strong momentum. Works in bull markets (breakouts continue) 
and bear markets (breakdowns continue). Uses weekly for structure, 6h for execution.
Target: 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_fractals(high, low):
    """Calculate Williams Fractals: bearish (peak) and bullish (turtle) fractals."""
    n = len(high)
    bearish = np.zeros(n, dtype=bool)
    bullish = np.zeros(n, dtype=bool)
    
    for i in range(2, n - 2):
        # Bearish fractal: high[i] is highest of 5 bars
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = True
        # Bullish fractal: low[i] is lowest of 5 bars
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = True
    
    return bearish, bullish

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for fractals
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly Williams Fractals (need 2 extra bars for confirmation)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    bearish_fractal, bullish_fractal = calculate_williams_fractals(high_1w, low_1w)
    # Williams fractal needs 2 extra weekly bars after the center bar for confirmation
    bearish_fractal_1w = align_htf_to_ltf(prices, df_1w, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_1w = align_htf_to_ltf(prices, df_1w, bullish_fractal.astype(float), additional_delay_bars=2)
    
    # Daily ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])) > 
                       (np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d), 
                       np.maximum(high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d) > 
                        (high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])), 
                        np.maximum(np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d, 0), 0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[0:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    adx_1d = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 6h volume / 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(bearish_fractal_1w[i]) or np.isnan(bullish_fractal_1w[i]) or 
            np.isnan(adx_1d[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        adx_val = adx_1d[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above bullish fractal (support) + ADX > 25 + volume surge
            if (bullish_fractal_1w[i] > 0.5 and 
                price_close > low_1w[-1] and  # Above most recent weekly low (fractal level)
                adx_val > 25 and 
                vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price below bearish fractal (resistance) + ADX > 25 + volume surge
            elif (bearish_fractal_1w[i] > 0.5 and 
                  price_close < high_1w[-1] and  # Below most recent weekly high (fractal level)
                  adx_val > 25 and 
                  vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: ADX falls below 20 (trend weakening) or opposite fractal touched
            if position == 1 and (adx_val < 20 or bearish_fractal_1w[i] > 0.5):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (adx_val < 20 or bullish_fractal_1w[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsFractal_Breakout_ADX_Volume"
timeframe = "6h"
leverage = 1.0