#!/usr/bin/env python3
# Hypothesis: 1h EMA crossover with 4h ADX trend filter and 1d volatility regime filter.
# Long when 1h EMA21 crosses above EMA50 AND 4h ADX > 25 (trending) AND 1d ATR ratio < 0.8 (low volatility).
# Short when 1h EMA21 crosses below EMA50 AND 4h ADX > 25 AND 1d ATR ratio < 0.8.
# Exit on opposite EMA crossover. Uses session filter (08-20 UTC) to avoid noise.
# Designed for 1h timeframe with strict entry to target 15-37 trades/year.
# Works in bull/bear: ADX filters strong trends, volatility filter avoids choppy regimes.

name = "1h_EMA21_50_ADX25_VolRegime_Session_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMAs on 1h
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h[0] = tr1[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    plus_dm = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h),
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)),
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_4h
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_4h
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Get 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR ratio: current ATR(7) / ATR(30) to detect low volatility regimes
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_7_1d = pd.Series(tr_1d).rolling(window=7, min_periods=7).mean().values
    atr_30_1d = pd.Series(tr_1d).rolling(window=30, min_periods=30).mean().values
    atr_ratio_1d = atr_7_1d / atr_30_1d
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # Already datetime64[ms], .hour works directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMAs
        # Skip if any required data is NaN
        if (np.isnan(ema21[i]) or np.isnan(ema50[i]) or 
            np.isnan(adx_4h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check session: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: EMA21 > EMA50 AND ADX > 25 AND ATR ratio < 0.8 (low vol)
            if ema21[i] > ema50[i] and adx_4h_aligned[i] > 25 and atr_ratio_1d_aligned[i] < 0.8:
                signals[i] = 0.20
                position = 1
            # SHORT: EMA21 < EMA50 AND ADX > 25 AND ATR ratio < 0.8 (low vol)
            elif ema21[i] < ema50[i] and adx_4h_aligned[i] > 25 and atr_ratio_1d_aligned[i] < 0.8:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: EMA21 crosses below EMA50
            if ema21[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: EMA21 crosses above EMA50
            if ema21[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals