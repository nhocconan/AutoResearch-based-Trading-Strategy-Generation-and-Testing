#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R with ADX regime filter
# Williams %R identifies overbought/oversold conditions (-20 to -80 range)
# ADX determines trend strength: ADX > 25 = trending, ADX < 20 = ranging
# In trending regime (ADX > 25): fade Williams %R extremes (short >-20, long <-80)
# In ranging regime (ADX < 20): momentum continuation (long >-20, short <-80)
# Uses 1d indicators aligned to 6h timeframe with proper delay
# Discrete position sizing 0.25 targets ~12-37 trades/year to minimize fee drag
# Works in bull/bear markets: mean reversion in trends, momentum in ranges adapts to all conditions

name = "6h_1d_williamsr_adx_regime_v1"
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
    
    # Calculate 1d Williams %R(14)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when high == low
        wr = np.where((highest_high - lowest_low) == 0, -50, wr)
        return wr
    
    williams_r_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d ADX(14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing
        def wilders_smoothing(values, period):
            if len(values) < period:
                return np.full(len(values), np.nan)
            alpha = 1.0 / period
            result = np.full(len(values), np.nan)
            result[period-1] = np.nanmean(values[:period])
            for i in range(period, len(values)):
                result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_dm_smoothed = wilders_smoothing(plus_dm, period)
        minus_dm_smoothed = wilders_smoothing(minus_dm, period)
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smoothed / atr
        minus_di = 100 * minus_dm_smoothed / atr
        
        # DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx = np.where((plus_di + minus_di) == 0, 0, dx)
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_1d_aligned[i]
        adx = adx_1d_aligned[i]
        
        if position == 1:  # Long position
            if adx > 25:  # Trending regime: mean reversion
                # Exit long if Williams %R rises above -20 (overbought)
                if wr > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime: momentum continuation
                # Exit long if Williams %R falls below -80 (oversold)
                if wr < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            if adx > 25:  # Trending regime: mean reversion
                # Exit short if Williams %R falls below -80 (oversold)
                if wr < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime: momentum continuation
                # Exit short if Williams %R rises above -20 (overbought)
                if wr > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if adx > 25:  # Trending regime: mean reversion at extremes
                if wr < -80:  # Oversold -> long
                    position = 1
                    signals[i] = 0.25
                elif wr > -20:  # Overbought -> short
                    position = -1
                    signals[i] = -0.25
            else:  # Ranging regime: momentum continuation
                if wr > -20:  # Overbought and rising -> long
                    position = 1
                    signals[i] = 0.25
                elif wr < -80:  # Oversold and falling -> short
                    position = -1
                    signals[i] = -0.25
    
    return signals