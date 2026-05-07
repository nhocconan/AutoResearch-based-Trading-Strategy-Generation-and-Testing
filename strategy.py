#!/usr/bin/env python3
# 4h_ADX_DMI_Trend_1dADX_Filter
# Hypothesis: ADX-based trend strength filter with DMI crossover on 4h timeframe, filtered by 1d ADX to avoid ranging markets.
# Long when +DI crosses above -DI with ADX > 25 on 4h and 1d ADX > 20 (trending market).
# Short when -DI crosses above +DI with ADX > 25 on 4h and 1d ADX > 20.
# Uses 14-period ADX/DMI. Avoids low-adx chop to reduce false signals and whipsaws.
# Target: 20-30 trades per year (~80-120 over 4 years) with position size 0.25.

name = "4h_ADX_DMI_Trend_1dADX_Filter"
timeframe = "4h"
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
    
    # Load 1-day data ONCE for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day ADX for regime filter (trending market)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and DM for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = tr2_1d[0] = tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    plus_dm_1d = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm_1d = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm_1d[0] = minus_dm_1d[0] = 0
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm_1d).ewm(alpha=1/14, adjust=False).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm_1d).ewm(alpha=1/14, adjust=False).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 4-hour ADX/DMI for entry signals
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = minus_dm[0] = 0
    
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need sufficient data for ADX calculation
    
    for i in range(start_idx, n):
        if np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine signals
        adx_trending = adx[i] > 25
        adx_1d_trending = adx_1d_aligned[i] > 20  # 1d must also be trending
        
        # DMI crossover signals
        bullish_cross = (plus_di[i] > minus_di[i]) and (plus_di[i-1] <= minus_di[i-1])
        bearish_cross = (minus_di[i] > plus_di[i]) and (minus_di[i-1] <= plus_di[i-1])
        
        if position == 0:
            # Long: bullish DMI crossover in trending market (both timeframes)
            if bullish_cross and adx_trending and adx_1d_trending:
                signals[i] = 0.25
                position = 1
            # Short: bearish DMI crossover in trending market (both timeframes)
            elif bearish_cross and adx_trending and adx_1d_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bearish DMI crossover or ADX drops below threshold
            if bearish_cross or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bullish DMI crossover or ADX drops below threshold
            if bullish_cross or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals