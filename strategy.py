#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: Weekly EMA200 trend + Daily Pivot Point (PP) breakout with volume confirmation
    # Uses weekly trend filter to avoid counter-trend trades, daily pivot levels for institutional relevance
    # Volume surge filters low-probability breakouts. Works in bull/bear via trend filter.
    
    # Load weekly and daily data once
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly EMA200 trend filter
    close_1w = df_1w['close'].values
    ema_1w_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # Daily Pivot Point calculation (standard formula)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pp_1d - low_1d
    s1_1d = 2 * pp_1d - high_1d
    
    # Align daily levels to 4h
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 4h ATR for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (20-period MA surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema_1w_200_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above S1 with volume surge AND weekly EMA200 uptrend
            if close[i] > s1_1d_aligned[i] and vol_surge[i] and close[i] > ema_1w_200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below R1 with volume surge AND weekly EMA200 downtrend
            elif close[i] < r1_1d_aligned[i] and vol_surge[i] and close[i] < ema_1w_200_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to pivot point level
            if position == 1:
                if close[i] < pp_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pp_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WeeklyEMA200_DailyPivot_PP_R1_S1_Breakout_VolumeSurge_v1"
timeframe = "4h"
leverage = 1.0