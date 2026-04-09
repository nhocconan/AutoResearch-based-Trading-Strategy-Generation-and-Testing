#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX trend filter + volume confirmation
# Williams %R(14) identifies overbought/oversold conditions on 6h chart
# 1d ADX > 25 confirms strong trend direction for breakout continuation
# Volume confirmation ensures breakouts have participation
# In strong trends (ADX > 25): trade breakouts in direction of trend
# In weak trends (ADX <= 25): avoid trading to prevent whipsaws
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: ADX filter adapts to trend strength, Williams %R captures reversals within trends

name = "6h_1d_williamsr_adx_volume_v1"
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
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed > 0, 100 * dm_plus_smoothed / tr_smoothed, 0)
    di_minus = np.where(tr_smoothed > 0, 100 * dm_minus_smoothed / tr_smoothed, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Calculate 6h Williams %R(14)
    def williams_r(high, low, close, period):
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        wr = np.where((highest_high - lowest_low) != 0, 
                      -100 * (highest_high - close) / (highest_high - lowest_low), 
                      -50)
        return wr
    
    wr_6h = williams_r(high, low, close, 14)
    
    # Calculate 6h average volume (20-period)
    avg_volume_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume_6h
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(wr_6h[i]) or 
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when 1d ADX > 25 (strong trend)
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit long if Williams %R rises above -20 (overbought) or trend weakens
            if wr_6h[i] > -20 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if Williams %R falls below -80 (oversold) or trend weakens
            if wr_6h[i] < -80 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when Williams %R < -80 (oversold) in strong uptrend with volume
            if wr_6h[i] < -80 and strong_trend and volume_confirmed[i]:
                position = 1
                signals[i] = 0.25
            # Enter short when Williams %R > -20 (overbought) in strong downtrend with volume
            elif wr_6h[i] > -20 and strong_trend and volume_confirmed[i]:
                position = -1
                signals[i] = -0.25
    
    return signals