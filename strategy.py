#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R + 1w ADX regime filter for mean reversion
# Williams %R (14) identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold
# Weekly ADX (14) detects regime: > 25 = trending, < 20 = ranging
# In ranging markets (ADX < 20): mean reversion - buy oversold (%R < -80), sell overbought (%R > -20)
# In trending markets (ADX > 25): trend continuation - buy on pullbacks (%R crosses above -50 from below),
#   sell on rallies (%R crosses below -50 from above)
# Designed for low trade frequency (target 10-20/year) with clear regime adaptation
# Works in both bull (trend following in uptrends) and bear (mean reversion in ranges) markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1w data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w Williams %R (14 period)
    highest_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1w) / (highest_high - lowest_low + 1e-10)
    
    # 1w ATR(14) for volatility normalization
    tr1 = np.maximum(high_1w[1:], low_1w[:-1]) - np.minimum(high_1w[1:], low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1w ADX(14) for regime detection
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to daily timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            continue
        
        # Regime: ADX > 25 = trending, ADX < 20 = ranging
        if adx_aligned[i] < 20:
            # Ranging market: mean reversion
            # Buy when oversold (%R < -80)
            if williams_r_aligned[i] < -80 and position <= 0:
                position = 1
                signals[i] = position_size
            # Sell when overbought (%R > -20)
            elif williams_r_aligned[i] > -20 and position >= 0:
                position = -1
                signals[i] = -position_size
            # Exit when %R returns to neutral zone (-50)
            elif position == 1 and williams_r_aligned[i] > -50:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r_aligned[i] < -50:
                position = 0
                signals[i] = 0.0
        else:
            # Trending market: trend continuation on pullbacks
            # Buy when %R crosses above -50 from below (pullback in uptrend)
            if williams_r_aligned[i] > -50 and williams_r_aligned[i-1] <= -50 and position <= 0:
                position = 1
                signals[i] = position_size
            # Sell when %R crosses below -50 from above (pullback in downtrend)
            elif williams_r_aligned[i] < -50 and williams_r_aligned[i-1] >= -50 and position >= 0:
                position = -1
                signals[i] = -position_size
            # Exit when %R reaches extreme opposite
            elif position == 1 and williams_r_aligned[i] < -80:
                position = 0
                signals[i] = 0.0
            elif position == -1 and williams_r_aligned[i] > -20:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1d_1w_WilliamsR_ADX_Regime"
timeframe = "1d"
leverage = 1.0