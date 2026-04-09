#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1d Williams %R extremes with 1w ADX regime filter
# Williams %R measures overbought/oversold: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# %R < -80 = oversold, %R > -20 = overbought
# ADX > 25 indicates trending market; ADX < 20 indicates ranging market
# In trending regime (ADX > 25): trade pullbacks (long when %R crosses above -80 from below, short when %R crosses below -20 from above)
# In ranging regime (ADX < 20): mean revert at extremes (long when %R < -80, short when %R > -20)
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: trend-following pullbacks in strong trends, mean reversion in ranging markets

name = "6h_1d_1w_williamsr_adx_regime_v1"
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
        williams_r = np.where((highest_high - lowest_low) != 0,
                              ((highest_high - close) / (highest_high - lowest_low)) * -100, -50)
        return williams_r
    
    williams_r_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    
    # Load 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14)
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
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
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 1w ADX
        trending_regime = adx_1w_aligned[i] > 25
        ranging_regime = adx_1w_aligned[i] < 20
        
        if position == 1:  # Long position
            if trending_regime:
                # Exit long if Williams %R rises above -20 (overbought)
                if williams_r_1d_aligned[i] >= -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif ranging_regime:
                # Exit long if Williams %R returns from oversold
                if williams_r_1d_aligned[i] > -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if trending_regime:
                # Exit short if Williams %R falls below -80 (oversold)
                if williams_r_1d_aligned[i] <= -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif ranging_regime:
                # Exit short if Williams %R returns from overbought
                if williams_r_1d_aligned[i] < -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if trending_regime:
                # Trade pullbacks in trending market
                if i > 100 and williams_r_1d_aligned[i-1] < -80 and williams_r_1d_aligned[i] >= -80:
                    # Williams %R crossed above -80 from below (end of pullback)
                    position = 1
                    signals[i] = 0.25
                elif i > 100 and williams_r_1d_aligned[i-1] > -20 and williams_r_1d_aligned[i] <= -20:
                    # Williams %R crossed below -20 from above (end of bounce)
                    position = -1
                    signals[i] = -0.25
            elif ranging_regime:
                # Mean revert at extremes in ranging market
                if williams_r_1d_aligned[i] <= -80:
                    position = 1
                    signals[i] = 0.25
                elif williams_r_1d_aligned[i] >= -20:
                    position = -1
                    signals[i] = -0.25
    
    return signals