#!/usr/bin/env python3
# 6h Monthly Pivot Bounce with Volume and Trend Filter
# Hypothesis: Monthly pivot points act as stronger institutional support/resistance than weekly.
# Price bouncing off monthly S1/S2/R1/R2 with volume > 1.3x 20-period average and ADX > 20
# captures institutional defense of key levels. Works in bull/bear by trading reversals
# from strong monthly levels with trend filter to avoid whipsaws in weak trends.
# Target: 10-25 trades/year per symbol.

name = "6h_monthly_pivot_bounce_volume_adx_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get monthly data for pivot points and ADX - call ONCE before loop
    df_m = get_htf_data(prices, '1M')
    high_m = df_m['high'].values
    low_m = df_m['low'].values
    close_m = df_m['close'].values
    volume_m = df_m['volume'].values
    
    # Calculate monthly pivot points (standard floor trader pivots)
    pp_m = (high_m + low_m + close_m) / 3
    r1_m = 2 * pp_m - low_m
    s1_m = 2 * pp_m - high_m
    r2_m = pp_m + (high_m - low_m)
    s2_m = pp_m - (high_m - low_m)
    
    # Calculate 20-period average volume for monthly timeframe
    vol_ma_m = pd.Series(volume_m).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX on monthly timeframe (trend strength filter)
    # True Range
    tr1 = high_m - low_m
    tr2 = np.abs(high_m - np.roll(close_m, 1))
    tr3 = np.abs(low_m - np.roll(close_m, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_m - np.roll(high_m, 1)
    down_move = np.roll(low_m, 1) - low_m
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Handle division by zero and invalid values
    adx = np.where((plus_di + minus_di) == 0, 0, adx)
    adx = np.where(np.isnan(adx) | np.isinf(adx), 0, adx)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30
    
    for i in range(start_idx, n):
        # Get aligned monthly values for current 6h bar
        r1 = align_htf_to_ltf(prices, df_m, r1_m)[i]
        s1 = align_htf_to_ltf(prices, df_m, s1_m)[i]
        r2 = align_htf_to_ltf(prices, df_m, r2_m)[i]
        s2 = align_htf_to_ltf(prices, df_m, s2_m)[i]
        vol_ma = align_htf_to_ltf(prices, df_m, vol_ma_m)[i]
        adx_val = align_htf_to_ltf(prices, df_m, adx)[i]
        
        # Skip if any required data is NaN
        if np.isnan(r1) or np.isnan(s1) or np.isnan(r2) or np.isnan(s2) or np.isnan(vol_ma) or np.isnan(adx_val) or volume[i] == 0:
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x 20-period average
        vol_condition = volume[i] > 1.3 * vol_ma
        
        # Trend condition: ADX > 20
        trend_condition = adx_val > 20
        
        if position == 1:  # Long position
            # Exit if price breaks below S2 (bounce failed)
            if close[i] < s2:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above R2 (bounce failed)
            if close[i] > r2:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bounce long at S1/S2 with volume and trend confirmation
            if ((low[i] <= s1 and close[i] > s1) or (low[i] <= s2 and close[i] > s2)) and vol_condition and trend_condition:
                position = 1
                signals[i] = 0.25
            # Bounce short at R1/R2 with volume and trend confirmation
            elif ((high[i] >= r1 and close[i] < r1) or (high[i] >= r2 and close[i] < r2)) and vol_condition and trend_condition:
                position = -1
                signals[i] = -0.25
    
    return signals