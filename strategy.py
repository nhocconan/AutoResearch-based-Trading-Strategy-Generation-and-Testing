#!/usr/bin/env python3
"""
12h_KAMA_Regime_Adaptive_v1
Hypothesis: Adaptive strategy using Kaufman Adaptive Moving Average (KAMA) for trend detection,
combined with choppiness index regime filter and volume confirmation. In trending regimes (CHOP < 38.2),
follow KAMA direction; in ranging regimes (CHOP > 61.8), mean-revert at Bollinger Bands.
Uses 1d EMA50 as higher timeframe trend filter and 1d ADX > 20 for trend strength.
Designed for low trade frequency (12-25/year) to work in both bull (2021-2023) and bear (2022, 2025+) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d ADX(14) for trend strength filter
    def calculate_adx(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed TR, DM+
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        dm_plus_sum = pd.Series(dm_plus).rolling(window=window, min_periods=window).sum().values
        dm_minus_sum = pd.Series(dm_minus).rolling(window=window, min_periods=window).sum().values
        
        # Directional Indicators
        tr_sum_safe = np.where(tr_sum == 0, 1e-10, tr_sum)
        di_plus = 100 * dm_plus_sum / tr_sum_safe
        di_minus = 100 * dm_minus_sum / tr_sum_safe
        
        # DX and ADX
        dx = np.zeros(len(close))
        dx_denom = di_plus + di_minus
        dx_denom_safe = np.where(dx_denom == 0, 1e-10, dx_denom)
        dx = 100 * np.abs(di_plus - di_minus) / dx_denom_safe
        
        adx = pd.Series(dx).rolling(window=window, min_periods=window).mean().values
        return adx
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, window=14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate KAMA (12h)
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=er_length))
        volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if er_length == 1 else \
                     pd.Series(close).rolling(window=er_length).apply(
                         lambda x: np.sum(np.abs(np.diff(x))), raw=True).values
        # Fix first er_length values
        volatility[:er_length] = np.nan
        for i in range(er_length, len(close)):
            volatility[i] = np.sum(np.abs(np.diff(close[i-er_length+1:i+1])))
        
        er = np.where(volatility > 0, change / volatility, 0)
        # Smoothing Constant
        sc = np.power(er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1), 2)
        # KAMA
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if np.isnan(sc[i]):
                kama[i] = kama[i-1]
            else:
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Calculate Choppiness Index (12h)
    def calculate_choppiness(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Sum of TR over window
        tr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Highest high and lowest low over window
        max_h = pd.Series(high).rolling(window=window, min_periods=window).max().values
        min_l = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        # Choppiness Index
        chop = np.zeros(len(close))
        denom = max_h - min_l
        denom_safe = np.where(denom == 0, 1e-10, denom)
        chop = 100 * np.log10(tr_sum / denom_safe) / np.log10(window)
        return chop
    
    chop = calculate_choppiness(high, low, close, window=14)
    
    # Calculate Bollinger Bands (12h, 20, 2)
    bb_period = 20
    bb_std = 2
    bb_ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + (bb_std_dev * bb_std)
    bb_lower = bb_ma - (bb_std_dev * bb_std)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for all indicators
    start_idx = max(50, bb_period, 20)  # EMA50 needs 50, BB needs 20, vol needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(kama[i]) or np.isnan(chop[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        df_1d_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(df_1d_close_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        htf_1d_bullish = df_1d_close_aligned[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = df_1d_close_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: moderate spike (vol_ratio > 1.5)
        volume_confirmed = vol_ratio[i] > 1.5
        
        # Trend strength filter: only trade when ADX > 20 (moderate trend)
        trend_filter = adx_1d_aligned[i] > 20.0
        
        # Regime filters
        is_trending = chop[i] < 38.2   # Trending regime
        is_ranging = chop[i] > 61.8    # Ranging regime
        
        if position == 0:
            # Long setup conditions
            if is_trending:
                # In trending regime: follow KAMA (price above KAMA = bullish)
                long_setup = (close[i] > kama[i]) and htf_1d_bullish and volume_confirmed and trend_filter
            elif is_ranging:
                # In ranging regime: mean revert at Bollinger Bands (price at lower band = long)
                long_setup = (close[i] <= bb_lower[i]) and htf_1d_bullish and volume_confirmed and trend_filter
            else:
                long_setup = False
            
            # Short setup conditions
            if is_trending:
                # In trending regime: follow KAMA (price below KAMA = bearish)
                short_setup = (close[i] < kama[i]) and htf_1d_bearish and volume_confirmed and trend_filter
            elif is_ranging:
                # In ranging regime: mean revert at Bollinger Bands (price at upper band = short)
                short_setup = (close[i] >= bb_upper[i]) and htf_1d_bearish and volume_confirmed and trend_filter
            else:
                short_setup = False
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions
            if is_trending:
                # Exit when price crosses below KAMA
                if close[i] < kama[i]:
                    signals[i] = 0.0
                    position = 0
            elif is_ranging:
                # Exit when price reaches middle Bollinger Band
                if close[i] >= bb_ma[i]:
                    signals[i] = 0.0
                    position = 0
            # Also exit if HTF trend turns bearish
            if not htf_1d_bullish:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if is_trending:
                # Exit when price crosses above KAMA
                if close[i] > kama[i]:
                    signals[i] = 0.0
                    position = 0
            elif is_ranging:
                # Exit when price reaches middle Bollinger Band
                if close[i] <= bb_ma[i]:
                    signals[i] = 0.0
                    position = 0
            # Also exit if HTF trend turns bullish
            if htf_1d_bullish:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Regime_Adaptive_v1"
timeframe = "12h"
leverage = 1.0