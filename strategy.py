#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray + ADX regime filter with 1d EMA50 trend alignment.
# Bull Power (high - EMA13) and Bear Power (EMA13 - low) measure buying/selling pressure.
# ADX > 25 indicates trending market; we trade only in the direction of the 1d EMA50 trend.
# Long when Bull Power > 0, ADX > 25, and close > 1d EMA50.
# Short when Bear Power > 0, ADX > 25, and close < 1d EMA50.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 6h timeframe.
# Designed to capture strong trending moves while avoiding choppy markets, working in both bull and bear regimes.

name = "6h_ElderRay_ADX_Regime_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    lookback_ema = 13
    ema_13 = pd.Series(close).ewm(span=lookback_ema, adjust=False, min_periods=lookback_ema).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Buying pressure
    bear_power = ema_13 - low   # Selling pressure
    
    # Calculate ADX (14) for regime filter
    lookback_adx = 14
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = pd.Series(tr).ewm(span=lookback_adx, adjust=False, min_periods=lookback_adx).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=lookback_adx, adjust=False, min_periods=lookback_adx).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=lookback_adx, adjust=False, min_periods=lookback_adx).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=lookback_adx, adjust=False, min_periods=lookback_adx).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_ema, lookback_adx, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (buying pressure), ADX > 25 (trending), close > 1d EMA50 (uptrend)
            if (bull_power[i] > 0 and 
                adx[i] > 25 and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 (selling pressure), ADX > 25 (trending), close < 1d EMA50 (downtrend)
            elif (bear_power[i] > 0 and 
                  adx[i] > 25 and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (loss of buying pressure) OR ADX <= 25 (losing trend)
            if (bull_power[i] <= 0 or adx[i] <= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power <= 0 (loss of selling pressure) OR ADX <= 25 (losing trend)
            if (bear_power[i] <= 0 or adx[i] <= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals