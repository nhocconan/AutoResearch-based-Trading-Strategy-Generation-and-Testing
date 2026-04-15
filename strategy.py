#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength
# ADX > 25 indicates trending market; < 20 indicates ranging
# In trending markets (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
# In ranging markets (ADX < 20): fade extremes - short when Bull Power > 0.5*ATR, long when Bear Power > 0.5*ATR
# Uses 12h EMA13 and ATR for Elder Ray, 12h ADX for regime
# Designed for low trade frequency (target 15-25/year) with clear regime adaptation
# Works in both bull (trend following) and bear (mean reversion in ranges) markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA13 for Elder Ray
    ema13 = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 12h ATR(14) for volatility normalization
    tr1 = np.maximum(high_12h[1:], low_12h[:-1]) - np.minimum(high_12h[1:], low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_12h - ema13
    bear_power = ema13 - low_12h
    
    # 12h ADX(14) for regime detection
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_14 + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_14 + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_12h, ema13)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            continue
        
        # Regime: ADX > 25 = trending, ADX < 20 = ranging
        if adx_aligned[i] > 25:
            # Trending market: follow Elder Ray momentum
            bull_rising = bull_power_aligned[i] > bull_power_aligned[i-1]
            bear_rising = bear_power_aligned[i] > bear_power_aligned[i-1]
            
            # Long when Bull Power positive and rising
            if bull_power_aligned[i] > 0 and bull_rising and position <= 0:
                position = 1
                signals[i] = position_size
            # Short when Bear Power positive and rising
            elif bear_power_aligned[i] > 0 and bear_rising and position >= 0:
                position = -1
                signals[i] = -position_size
            # Exit when momentum fades
            elif position == 1 and not bull_rising:
                position = 0
                signals[i] = 0.0
            elif position == -1 and not bear_rising:
                position = 0
                signals[i] = 0.0
        else:
            # Ranging market: fade Elder Ray extremes
            # Short when Bull Power is excessively high (overbought)
            if bull_power_aligned[i] > 0.5 * atr_aligned[i] and position >= 0:
                position = -1
                signals[i] = -position_size
            # Long when Bear Power is excessively high (oversold)
            elif bear_power_aligned[i] > 0.5 * atr_aligned[i] and position <= 0:
                position = 1
                signals[i] = position_size
            # Exit when power returns to neutral zone
            elif position == 1 and bear_power_aligned[i] <= 0.2 * atr_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and bull_power_aligned[i] <= 0.2 * atr_aligned[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_12h_ElderRay_ADX_Regime"
timeframe = "6h"
leverage = 1.0