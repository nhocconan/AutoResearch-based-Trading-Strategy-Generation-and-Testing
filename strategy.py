#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d ADX regime filter
# In low ADX (ADX < 20) ranging markets: mean reversion at extreme Williams %R levels (<10 or >90)
# In high ADX (ADX > 25) trending markets: fade extreme Williams %R as continuation signal
# Uses 6h Williams %R(14) and 1d ADX(14) to avoid whipsaws in both bull and bear markets
# Discrete position sizing 0.25 targets ~12-37 trades/year to minimize fee drag

name = "6h_1d_williamsr_adx_regime_v1"
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
    
    # Calculate 1d ADX(14) - measures trend strength
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
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
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
    
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d > 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d > 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h Williams %R(14) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0,
                          -100 * (highest_high - close) / (highest_high - lowest_low),
                          -50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        wr = williams_r[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx > 25:  # Trending market
                # Exit long if Williams %R rises above -50 (momentum fading)
                if wr > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Ranging market (ADX <= 25)
                # Exit long if Williams %R rises above -10 (overbought)
                if wr > -10:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if adx > 25:  # Trending market
                # Exit short if Williams %R falls below -50 (momentum fading)
                if wr < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Ranging market (ADX <= 25)
                # Exit short if Williams %R falls below -90 (oversold)
                if wr < -90:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry conditions
            if adx > 25:  # Trending market
                # Enter short on extreme overbought (continuation fade)
                if wr < -90:
                    position = -1
                    signals[i] = -0.25
                # Enter long on extreme oversold (continuation fade)
                elif wr > -10:
                    position = 1
                    signals[i] = 0.25
            else:  # Ranging market (ADX <= 25)
                # Mean reversion: buy oversold, sell overbought
                if wr <= -90:
                    position = 1
                    signals[i] = 0.25
                elif wr >= -10:
                    position = -1
                    signals[i] = -0.25
    
    return signals