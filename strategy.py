#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX trend filter + volume confirmation
# In strong trends (1d ADX > 25): Williams %R mean reversion (long < -80, short > -20)
# In weak trends/ranging (1d ADX <= 25): Williams %R trend following (long > -20 pullback, short < -80 bounce)
# Uses discrete position sizing 0.25 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: adapts to trend strength via ADX regime filter

name = "6h_1d_williamsr_adx_volume_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate 1d ATR(14) for ADX
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
    
    # Calculate 1d +DM and -DM
    up_move = np.concatenate([[np.nan], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1] - low_1d[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Calculate 1d +DI and -DI
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / atr_1d
    
    # Calculate 1d ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Calculate 6h Williams %R (14-period)
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_6h = -100 * (highest_high_6h - close) / (highest_high_6h - lowest_low_6h)
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d)
    
    # Pre-compute volume confirmation array (6h volume > 1.5x 20-period average)
    avg_volume_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.5 * avg_volume_6h
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r_6h[i]) or
            np.isnan(plus_di_1d_aligned[i]) or np.isnan(minus_di_1d_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter based on 1d ADX
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] <= 25
        
        if position == 1:  # Long position
            if strong_trend:
                # Exit long if Williams %R rises above -20 (overbought) or trend weakens
                if williams_r_6h[i] > -20 or weak_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif weak_trend:
                # Exit long if Williams %R falls below -80 (oversold) or price makes new low
                if williams_r_6h[i] < -80 or low[i] < lowest_low_6h[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if strong_trend:
                # Exit short if Williams %R falls below -80 (oversold) or trend weakens
                if williams_r_6h[i] < -80 or weak_trend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            elif weak_trend:
                # Exit short if Williams %R rises above -20 (overbought) or price makes new high
                if williams_r_6h[i] > -20 or high[i] > highest_high_6h[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if strong_trend:
                # Enter long on Williams %R oversold (< -80) with volume confirmation
                if williams_r_6h[i] < -80 and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on Williams %R overbought (> -20) with volume confirmation
                elif williams_r_6h[i] > -20 and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif weak_trend:
                # Enter long on Williams %R pullback from oversold (> -80 and rising) with volume confirmation
                if williams_r_6h[i] > -80 and williams_r_6h[i] > williams_r_6h[i-1] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on Williams %R bounce from overbought (< -20 and falling) with volume confirmation
                elif williams_r_6h[i] < -20 and williams_r_6h[i] < williams_r_6h[i-1] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals