#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour 1-week Pivot Point (R1/S1) breakout with volume confirmation and ADX trend filter
# In trending markets (ADX > 25), trade breakouts of weekly pivot levels with volume confirmation
# In ranging markets (ADX < 20), trade mean reversion at weekly pivot levels with volume confirmation
# Weekly pivot levels provide strong support/resistance; volume confirms conviction; ADX filters regime
# Target: 15-30 trades/year to minimize fee decay while capturing high-probability moves

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Pivot Points (R1, S1, Pivot) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot points: P = (H + L + C)/3, R1 = 2*P - L, S1 = 2*P - H
    pivot = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pivot - low_1w
    s1 = 2 * pivot - high_1w
    
    # Align weekly pivot levels to 12h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Daily ADX (14-period) for trend/range regime ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === Daily Volume Spike (vs 20-period average) ===
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 12h volume (avoid calling get_htf_data in loop)
        volume_12h_aligned = align_htf_to_ltf(prices, volume)
        
        # Volume spike: current 12h volume > 1.5x 20-period average
        vol_spike = volume_12h_aligned[i] > vol_ma_20_aligned[i] * 1.5
        
        # Regime filters
        is_trending = adx_aligned[i] > 25   # Trending market
        is_ranging = adx_aligned[i] < 20    # Ranging market
        
        # Entry logic: only enter when flat
        if position == 0:
            # In trending markets: trade breakouts of weekly pivot levels with volume confirmation
            if is_trending:
                breakout_up = close[i] > r1_aligned[i-1]  # Break above R1
                breakout_down = close[i] < s1_aligned[i-1]  # Break below S1
                
                if breakout_up and vol_spike:
                    signals[i] = 0.25
                    position = 1
                    continue
                elif breakout_down and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    continue
            
            # In ranging markets: trade mean reversion at weekly pivot levels with volume confirmation
            elif is_ranging:
                # Mean reversion: price touches S1 (support) or R1 (resistance) with volume spike
                if close[i] <= s1_aligned[i] and vol_spike:
                    signals[i] = 0.25  # Buy at S1 (support)
                    position = 1
                    continue
                elif close[i] >= r1_aligned[i] and vol_spike:
                    signals[i] = -0.25  # Sell at R1 (resistance)
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long based on regime
            if is_trending:
                # In trending market: exit when price returns to weekly pivot
                if close[i] < pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = 0.25
            else:  # ranging market
                # In ranging market: exit when price returns to weekly pivot
                if close[i] >= pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit short based on regime
            if is_trending:
                # In trending market: exit when price returns to weekly pivot
                if close[i] > pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = -0.25
            else:  # ranging market
                # In ranging market: exit when price returns to weekly pivot
                if close[i] <= pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_R1S1_ADXVolBreakout"
timeframe = "12h"
leverage = 1.0