#!/usr/bin/env python3
# 6h_elder_ray_regime_v2
# Hypothesis: 6h strategy using Elder Ray (Bull/Bear Power) with EMA13 trend filter and 1d HTF regime filter (ADX>25). 
# Enters long when Bull Power > 0 and Bear Power < 0 with EMA13 bullish alignment and 1d ADX>25 (trending market).
# Enters short when Bear Power > 0 and Bull Power < 0 with EMA13 bearish alignment and 1d ADX>25.
# Uses discrete position sizing (0.25) to limit fee drag. Designed for low turnover (target: 12-37 trades/year) 
# to work in both bull and bear markets by trading with the trend only when higher timeframe confirms strong trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    if len(high) < period + 1:
        return np.full_like(high, np.nan, dtype=float)
    
    # True Range
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift(1)).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    # Directional Movement
    up_move = pd.Series(high).diff()
    down_move = -pd.Series(low).diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean() / atr
    
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    
    return adx.values

name = "6h_elder_ray_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA13 for trend filter on primary timeframe
    ema_13 = calculate_ema(close, 13)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # 1d HTF regime filter: ADX(14)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    adx_14_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 1d ADX > 25 (trending market)
        trending_market = adx_14_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: Bear Power becomes positive (bulls losing control) OR EMA13 bearish flip
            if bear_power[i] > 0 or close[i] < ema_13[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power becomes positive (bears losing control) OR EMA13 bullish flip
            if bull_power[i] > 0 or close[i] > ema_13[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter only with Elder Ray signals and regime filter
            if trending_market:
                # Long: Bull Power positive AND Bear Power negative (bulls in control) with EMA13 bullish
                if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_13[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: Bear Power positive AND Bull Power negative (bears in control) with EMA13 bearish
                elif bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema_13[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals