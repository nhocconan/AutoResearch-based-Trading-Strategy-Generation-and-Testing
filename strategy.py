#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h ADX trend filter + volume confirmation
# Williams %R identifies overbought/oversold conditions; ADX confirms trend strength
# In strong trends (ADX > 25): fade extreme %R readings (mean reversion within trend)
# In weak trends (ADX <= 25): avoid trading to prevent whipsaws
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: %R captures reversals, ADX filter avoids sideways chop

name = "6h_12h_williamsr_adx_volume_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Williams %R (14-period)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r_12h = np.where(
        (highest_high_12h - lowest_low_12h) != 0,
        ((highest_high_12h - close_12h) / (highest_high_12h - lowest_low_12h)) * -100,
        -50  # neutral when range is zero
    )
    
    # Calculate 12h ADX (14-period) for trend strength
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    # True Range components
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr_12h = wilders_smoothing(tr_12h, 14)
    dm_plus_smoothed = wilders_smoothing(dm_plus, 14)
    dm_minus_smoothed = wilders_smoothing(dm_minus, 14)
    
    # Directional Indicators
    di_plus = np.where(atr_12h > 0, (dm_plus_smoothed / atr_12h) * 100, 0)
    di_minus = np.where(atr_12h > 0, (dm_minus_smoothed / atr_12h) * 100, 0)
    
    # ADX calculation
    dx = np.where((di_plus + di_minus) > 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_12h = wilders_smoothing(dx, 14)
    
    # Align 12h indicators to 6h timeframe
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in strong trends (ADX > 25)
        strong_trend = adx_12h_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit long if %R rises above -20 (overbought) or trend weakens
            if williams_r_12h_aligned[i] > -20 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if %R falls below -80 (oversold) or trend weakens
            if williams_r_12h_aligned[i] < -80 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when %R is oversold (< -80) in strong trend with volume confirmation
            if williams_r_12h_aligned[i] < -80 and strong_trend and volume_confirmed[i]:
                position = 1
                signals[i] = 0.25
            # Enter short when %R is overbought (> -20) in strong trend with volume confirmation
            elif williams_r_12h_aligned[i] > -20 and strong_trend and volume_confirmed[i]:
                position = -1
                signals[i] = -0.25
    
    return signals