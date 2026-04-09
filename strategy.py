#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R with volume confirmation and ADX trend filter
# Williams %R identifies overbought/oversold conditions; long when %R crosses above -80 from below,
# short when %R crosses below -20 from above, only in trending markets (ADX > 25).
# In ranging markets (ADX < 20), fade extremes: long at %R < -90, short at %R > -10.
# Uses discrete position sizing 0.25 to target ~30-60 trades/year and minimize fee drag.
# Works in bull/bear markets: momentum follows trends in trending regimes, mean reversion at extremes in ranging regimes.

name = "4h_1d_williamsr_adx_volume_v1"
timeframe = "4h"
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
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.zeros_like(close_1d)
    
    # Calculate 1d Williams %R(14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0, williams_r, -50)
    
    # Calculate 1d ADX(14) for trend regime filter
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_confirmed = not np.isnan(vol_ma_20[i]) and volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            if adx_1d_aligned[i] > 25:  # Trending regime
                # Exit long if Williams %R falls below -50 (momentum loss)
                if williams_r_aligned[i] < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging regime (ADX < 25)
                # Exit long if price moves back above Williams %R = -20 (overbought)
                if williams_r_aligned[i] > -20:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            if adx_1d_aligned[i] > 25:  # Trending regime
                # Exit short if Williams %R rises above -50 (momentum loss)
                if williams_r_aligned[i] > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging regime (ADX < 25)
                # Exit short if price moves back below Williams %R = -80 (oversold)
                if williams_r_aligned[i] < -80:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if adx_1d_aligned[i] > 25:  # Trending regime
                # Momentum strategy: long when %R crosses above -80, short when %R crosses below -20
                if williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and volume_confirmed:
                    position = 1
                    signals[i] = 0.25
                elif williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and volume_confirmed:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
            else:  # Ranging regime (ADX < 25)
                # Mean reversion: long at extreme oversold, short at extreme overbought
                if williams_r_aligned[i] < -90:
                    position = 1
                    signals[i] = 0.25
                elif williams_r_aligned[i] > -10:
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
    
    return signals