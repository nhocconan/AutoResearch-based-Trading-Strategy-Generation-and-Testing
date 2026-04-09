#!/usr/bin/env python3
# 6h_elder_ray_regime_v1
# Hypothesis: 6h Elder Ray (Bull/Bear Power) combined with 1d regime filter (ADX>25 for trending, ADX<20 for ranging). In trending regimes: long when Bull Power > 0 and Bear Power < 0, short when Bull Power < 0 and Bear Power > 0. In ranging regimes: mean reversion at Bollinger Band extremes (20,2) with Elder Ray divergence. Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 12-37 trades/year) to work in both bull and bear markets by adapting to volatility regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Elder Ray components
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX for regime detection (trending vs ranging)
    # Calculate True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['close'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate +DM and -DM
    up_move = pd.Series(df_1d['high']).diff()
    down_move = pd.Series(df_1d['low']).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    
    # Bollinger Bands for ranging regime (using 6h data)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2.0 * std20
    lower_band = sma20 - 2.0 * std20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema13_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        price = close[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx_val > 25:  # Trending regime exit: Bear Power turns positive
                if bear_val > 0:
                    position = 0
                    signals[i] = 0.0
            else:  # Ranging regime exit: price returns to mean (EMA13) or hits opposite band
                if price >= ema13_aligned[i] or price <= lower_band[i]:
                    position = 0
                    signals[i] = 0.0
            if position == 1:  # Still long
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if adx_val > 25:  # Trending regime exit: Bull Power turns negative
                if bull_val < 0:
                    position = 0
                    signals[i] = 0.0
            else:  # Ranging regime exit: price returns to mean (EMA13) or hits opposite band
                if price <= ema13_aligned[i] or price >= upper_band[i]:
                    position = 0
                    signals[i] = 0.0
            if position == -1:  # Still short
                signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if adx_val > 25:  # Trending regime
                # Long: Bull Power positive AND Bear Power negative (strong uptrend)
                if bull_val > 0 and bear_val < 0:
                    position = 1
                    signals[i] = 0.25
                # Short: Bull Power negative AND Bear Power positive (strong downtrend)
                elif bull_val < 0 and bear_val > 0:
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging regime (ADX < 25)
                # Long: price at lower Bollinger Band with Bull Power turning up
                if price <= lower_band[i] and bull_val > bear_val and bull_val > 0:
                    position = 1
                    signals[i] = 0.25
                # Short: price at upper Bollinger Band with Bear Power turning down
                elif price >= upper_band[i] and bear_val > bull_val and bear_val < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals