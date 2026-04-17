#!/usr/bin/env python3
"""
1d_WeeklyDonchian_Breakout_Volume_Regime_v1
Daily Donchian channel breakout with weekly trend filter, volume confirmation,
and chop regime filter. Long when price breaks above weekly Donchian high in trending market,
short when breaks below weekly Donchian low in trending market. Uses volume spike
to confirm breakout strength. Designed for 1d timeframe to capture multi-week trends.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily ATR(14) for stop and filters ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Weekly Donchian channels (20-period) ===
    df_1w = get_htf_data(prices, '1w')
    donch_high = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # === Weekly ADX(14) for trend strength ===
    # Calculate ADX on weekly data
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # True Range for weekly
    w_tr1 = w_high - w_low
    w_tr2 = np.abs(w_high - np.roll(w_close, 1))
    w_tr3 = np.abs(w_low - np.roll(w_close, 1))
    w_tr = np.maximum(w_tr1, np.maximum(w_tr2, w_tr3))
    w_tr[0] = w_tr1[0]
    w_atr = pd.Series(w_tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    w_plus_dm = np.where((w_high[1:] - w_high[:-1]) > (w_low[:-1] - w_low[1:]), 
                         np.maximum(w_high[1:] - w_high[:-1], 0), 0)
    w_minus_dm = np.where((w_low[:-1] - w_low[1:]) > (w_high[1:] - w_high[:-1]), 
                          np.maximum(w_low[:-1] - w_low[1:], 0), 0)
    w_plus_dm = np.concatenate([[0], w_plus_dm])
    w_minus_dm = np.concatenate([[0], w_minus_dm])
    
    w_plus_di = 100 * pd.Series(w_plus_dm).rolling(window=14, min_periods=14).sum().values / (w_atr * 14)
    w_minus_di = 100 * pd.Series(w_minus_dm).rolling(window=14, min_periods=14).sum().values / (w_atr * 14)
    w_dx = 100 * np.abs(w_plus_di - w_minus_di) / (w_plus_di + w_minus_di + 1e-10)
    w_adx = pd.Series(w_dx).rolling(window=14, min_periods=14).mean().values
    w_adx_aligned = align_htf_to_ltf(prices, df_1w, w_adx)
    
    # === Weekly Chopiness Index (14) for regime filter ===
    # Chop = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    w_sum_atr = pd.Series(w_atr).rolling(window=14, min_periods=14).sum().values
    w_max_high = pd.Series(w_high).rolling(window=14, min_periods=14).max().values
    w_min_low = pd.Series(w_low).rolling(window=14, min_periods=14).min().values
    w_range = w_max_high - w_min_low
    # Avoid division by zero
    w_range_safe = np.where(w_range == 0, 1e-10, w_range)
    w_chop = 100 * np.log10(w_sum_atr / w_range_safe) / np.log10(14)
    w_chop_aligned = align_htf_to_ltf(prices, df_1w, w_chop)
    
    # === Volume spike detector (daily) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(w_adx_aligned[i]) or 
            np.isnan(w_chop_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Volatility filter: avoid extremely low volatility days
            if atr[i] < 0.5 * np.nanmedian(atr[max(0, i-50):i+1]):
                signals[i] = 0.0
                continue
                
            # Long: price breaks above weekly Donchian high, strong trend, not choppy, volume spike
            if (close[i] > donch_high_aligned[i] and 
                w_adx_aligned[i] > 25 and 
                w_chop_aligned[i] < 61.8 and  # Trending regime (chop < 61.8)
                vol_ratio[i] > 1.5):          # Volume spike
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below weekly Donchian low, strong trend, not choppy, volume spike
            elif (close[i] < donch_low_aligned[i] and 
                  w_adx_aligned[i] > 25 and 
                  w_chop_aligned[i] < 61.8 and  # Trending regime
                  vol_ratio[i] > 1.5):          # Volume spike
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low OR trend weakens OR chop increases
            if (close[i] < donch_low_aligned[i] or 
                w_adx_aligned[i] < 20 or 
                w_chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high OR trend weakens OR chop increases
            if (close[i] > donch_high_aligned[i] or 
                w_adx_aligned[i] < 20 or 
                w_chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_Regime_v1"
timeframe = "1d"
leverage = 1.0