#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_ChopRegime_CloseOnly
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA34 trend filter and choppiness regime (<38.2) for strong trends. 
Uses close-only breakouts to reduce whipsaw. Designed for low trade frequency (<30/year) via tight regime and trend filters. 
Works in bull/bear via 1d trend alignment - only takes longs in uptrends, shorts in downtrends.
"""

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Camarilla levels from previous 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Choppiness index (14) - only trade in strong trends (chop < 38.2)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (hh14 - ll14)) / np.log10(14)
    chop_filter = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 34 for EMA, 14 for ATR/chop
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(trend_1d[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Only trade in strong trending regimes
        if not chop_filter[i]:
            # In choppy markets, go flat
            signals[i] = 0.0
            position = 0
            continue
            
        if position == 0:
            # Long: Close above Camarilla R3 AND 1d uptrend
            if close[i] > camarilla_r3_aligned[i] and trend_1d[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Close below Camarilla S3 AND 1d downtrend
            elif close[i] < camarilla_s3_aligned[i] and trend_1d[i] == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Close below Camarilla S3 OR 1d trend turns down
            if close[i] < camarilla_s3_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Close above Camarilla R3 OR 1d trend turns up
            if close[i] > camarilla_r3_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_ChopRegime_CloseOnly"
timeframe = "4h"
leverage = 1.0