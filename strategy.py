#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1d Elder Ray (Bull/Bear Power) with 1w ADX regime filter
# Elder Ray measures bull/bear power via EMA(13): Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# ADX > 25 indicates trending market; ADX < 20 indicates ranging market
# In trending regime (ADX > 25): follow Elder Ray signals (long when Bull Power > 0, short when Bear Power < 0)
# In ranging regime (ADX < 20): mean revert at extreme Elder Ray levels (long when Bull Power < -threshold, short when Bear Power > threshold)
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: trend following in strong trends, mean reversion in ranging markets

name = "6h_1d_1w_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(13) for Elder Ray
    close_s_1d = pd.Series(close_1d)
    ema13_1d = close_s_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA(13)
    bear_power_1d = low_1d - ema13_1d   # Bear Power = Low - EMA(13)
    
    # Calculate 1d ATR(14) for volatility normalization of Elder Ray
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    
    # Normalize Elder Ray by ATR to make it comparable across volatility regimes
    norm_bull_power_1d = np.where(atr_1d > 0, bull_power_1d / atr_1d, 0)
    norm_bear_power_1d = np.where(atr_1d > 0, bear_power_1d / atr_1d, 0)
    
    # Load 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
    def calculate_dmi(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[:-1])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr = wilders_smoothing(tr, period)
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed DM
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / atr
        minus_di = 100 * minus_dm_smooth / atr
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smoothing(dx, period)
        
        return adx
    
    adx_1w = calculate_dmi(high_1w, low_1w, close_1w, 14)
    
    # Align 1d indicators to 6h timeframe
    norm_bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, norm_bull_power_1d)
    norm_bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, norm_bear_power_1d)
    
    # Align 1w ADX to 6h timeframe
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Threshold for extreme Elder Ray levels in ranging market
    extreme_threshold = 1.5  # 1.5 ATR deviations
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(norm_bull_power_1d_aligned[i]) or np.isnan(norm_bear_power_1d_aligned[i]) or
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 1w ADX
        trending_regime = adx_1w_aligned[i] > 25
        ranging_regime = adx_1w_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if Bull Power turns negative
                if norm_bull_power_1d_aligned[i] <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if Bull Power returns from extreme
                if norm_bull_power_1d_aligned[i] > -extreme_threshold * 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if Bear Power turns positive
                if norm_bear_power_1d_aligned[i] >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if Bear Power returns from extreme
                if norm_bear_power_1d_aligned[i] < extreme_threshold * 0.5:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Follow Elder Ray signals in trending market
                if norm_bull_power_1d_aligned[i] > 0:
                    position = 1
                    signals[i] = 0.25
                elif norm_bear_power_1d_aligned[i] < 0:
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at extreme Elder Ray levels in ranging market
                if norm_bull_power_1d_aligned[i] < -extreme_threshold:
                    position = 1
                    signals[i] = 0.25
                elif norm_bear_power_1d_aligned[i] > extreme_threshold:
                    position = -1
                    signals[i] = -0.25
    
    return signals