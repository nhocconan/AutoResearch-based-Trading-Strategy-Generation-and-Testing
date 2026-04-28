#!/usr/bin/env python3
"""
4h_Vortex_Trend_With_Volume_Regime_Filter
Hypothesis: Uses Vortex Indicator (VI+) and (VI-) for trend direction, combined with volume spike (>2x 48-bar average) and Choppiness Index regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) to avoid whipsaws. Trades only in trending regimes with volume confirmation. Designed for low trade frequency (15-40/year) to minimize fee drag while capturing sustained trends. Works in both bull and bear by following Vortex crossover signals only when aligned with higher-timeframe trend (12h EMA50).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Vortex Indicator (VI+ and VI-) over 14 periods
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # VM+ and VM-
    vm_plus = np.abs(high - low[:-1])
    vm_minus = np.abs(low - high[:-1])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    n_periods = 14
    tr_sum = pd.Series(tr).rolling(window=n_periods, min_periods=n_periods).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=n_periods, min_periods=n_periods).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=n_periods, min_periods=n_periods).sum().values
    
    # VI+ and VI-
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Choppiness Index (CHOP) over 14 periods
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * n)) / log10(n)
    atr = pd.Series(tr).rolling(window=n_periods, min_periods=n_periods).mean().values
    chop_sum = pd.Series(tr).rolling(window=n_periods, min_periods=n_periods).sum().values
    chop = 100 * np.log10(chop_sum / (atr * n_periods)) / np.log10(n_periods)
    
    # Volume confirmation: >2x 48-period MA (8 days of 4h bars)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, n_periods)  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vi_plus[i]) or
            np.isnan(vi_minus[i]) or
            np.isnan(chop[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Vortex crossover signals
        vi_cross_up = vi_plus[i] > vi_minus[i] and vi_plus[i-1] <= vi_minus[i-1]
        vi_cross_down = vi_minus[i] > vi_plus[i] and vi_minus[i-1] <= vi_plus[i-1]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_48[i])
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        trending_regime = chop[i] < 38.2
        
        # Entry conditions
        long_entry = vi_cross_up and vol_confirm and trending_regime and uptrend
        short_entry = vi_cross_down and vol_confirm and trending_regime and downtrend
        
        # Exit conditions: opposite Vortex crossover or loss of trend
        long_exit = vi_cross_down or not uptrend
        short_exit = vi_cross_up or not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Vortex_Trend_With_Volume_Regime_Filter"
timeframe = "4h"
leverage = 1.0