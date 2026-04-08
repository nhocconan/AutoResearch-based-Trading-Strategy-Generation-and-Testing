#!/usr/bin/env python3
# 12h_1d_camellia_pivot_breakout_volume_v1
# Hypothesis: Daily Camarilla pivot levels act as key support/resistance. Price breaking above R4 or below S4 with volume > 1.5x 20-period average and ADX > 25 indicates institutional breakout. Works in bull/bear by capturing strong momentum moves from key daily levels with trend filter to avoid false signals in weak trends. Target: 12-37 trades/year per symbol.

name = "12h_1d_camellia_pivot_breakout_volume_v1"
timeframe = "12h"
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
    
    # Get daily data for Camarilla pivots and ADX - call ONCE before loop
    df_d = get_htf_data(prices, '1d')
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    volume_d = df_d['volume'].values
    
    # Calculate daily Camarilla pivot levels
    # Camarilla: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    # Where C = (H+L+C)/3 (typical price)
    typical_price = (high_d + low_d + close_d) / 3
    r4 = typical_price + ((high_d - low_d) * 1.1 / 2)
    s4 = typical_price - ((high_d - low_d) * 1.1 / 2)
    
    # Calculate 20-period average volume for daily timeframe
    vol_ma_d = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ADX on daily timeframe (trend strength filter)
    # True Range
    tr1 = high_d - low_d
    tr2 = np.abs(high_d - np.roll(close_d, 1))
    tr3 = np.abs(low_d - np.roll(close_d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_d - np.roll(high_d, 1)
    down_move = np.roll(low_d, 1) - low_d
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
        # Get aligned daily values for current 12h bar
        r4_level = align_htf_to_ltf(prices, df_d, r4)[i]
        s4_level = align_htf_to_ltf(prices, df_d, s4)[i]
        vol_ma = align_htf_to_ltf(prices, df_d, vol_ma_d)[i]
        adx_val = align_htf_to_ltf(prices, df_d, adx)[i]
        
        # Skip if any required data is NaN
        if np.isnan(r4_level) or np.isnan(s4_level) or np.isnan(vol_ma) or np.isnan(adx_val) or volume[i] == 0:
            signals[i] = 0.0
            continue
        
        # Volume breakout condition: current volume > 1.5x 20-period average
        vol_breakout = volume[i] > 1.5 * vol_ma
        
        # Strong trend condition: ADX > 25
        strong_trend = adx_val > 25
        
        if position == 1:  # Long position
            # Exit if price breaks below S4 (breakout failed)
            if close[i] < s4_level:
                position = 0
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above R4 (breakout failed)
            if close[i] > r4_level:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above R4 with volume confirmation and strong trend
            if high[i] >= r4_level and close[i] > r4_level and vol_breakout and strong_trend:
                position = 1
                signals[i] = 0.25
            # Breakout short below S4 with volume confirmation and strong trend
            elif low[i] <= s4_level and close[i] < s4_level and vol_breakout and strong_trend:
                position = -1
                signals[i] = -0.25
    
    return signals