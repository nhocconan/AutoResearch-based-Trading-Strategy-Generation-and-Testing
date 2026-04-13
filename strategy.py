#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation.
    # Long when price breaks above R4 with 1d ADX>25 and volume>1.5x average.
    # Short when price breaks below S4 with 1d ADX>25 and volume>1.5x average.
    # Exit when price returns to daily pivot point (PP).
    # Uses institutional pivot levels filtered by trend strength to avoid false breakouts.
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for pivot points and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # PP = (H + L + C) / 3
    # R4 = PP + ((H - L) * 1.1 / 2)
    # S4 = PP - ((H - L) * 1.1 / 2)
    pivot_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r4 = pivot_pp + ((high_1d - low_1d) * 1.1 / 2.0)
    camarilla_s4 = pivot_pp - ((high_1d - low_1d) * 1.1 / 2.0)
    
    # Calculate ADX(14) on 1d for trend filter
    # TR = max(H-L, abs(H-PC), abs(L-PC))
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    # Prepend first values to maintain length
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth TR, +DM, -DM with Wilder's smoothing (alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    pivot_pp_aligned = align_htf_to_ltf(prices, df_1d, pivot_pp)
    
    # Calculate volume average (20-period) on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(pivot_pp_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        long_breakout = (close[i] > camarilla_r4_aligned[i]) and strong_trend and volume_confirm
        short_breakout = (close[i] < camarilla_s4_aligned[i]) and strong_trend and volume_confirm
        
        # Exit conditions: price returns to daily pivot point
        long_exit = close[i] < pivot_pp_aligned[i]
        short_exit = close[i] > pivot_pp_aligned[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_breakout_adx_volume_v1"
timeframe = "6h"
leverage = 1.0