#!/usr/bin/env python3
# 12h_camarilla_1d_volume_chop_v1
# Hypothesis: 12h strategy using Camarilla pivot levels from 1d timeframe for entry/exit,
# volume confirmation, and choppiness regime filter to avoid whipsaws. Designed for low
# trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag.
# Works in bull/bear by using Camarilla levels as dynamic support/resistance and
# choppiness filter to only trade in ranging markets (CHOP > 61.8) or strong trends
# (CHOP < 38.2) with breakout logic. Uses discrete sizing (±0.25) to minimize fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots and choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # R2 = C + (H - L) * 1.1/6
    # R3 = C + (H - L) * 1.1/4
    # R4 = C + (H - L) * 1.1/2
    # S1 = C - (H - L) * 1.1/12
    # S2 = C - (H - L) * 1.1/6
    # S3 = C - (H - L) * 1.1/4
    # S4 = C - (H - L) * 1.1/2
    
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    r2_1d = close_1d + (high_1d - low_1d) * 1.1 / 6.0
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0
    r4_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    s2_1d = close_1d - (high_1d - low_1d) * 1.1 / 6.0
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0
    s4_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate choppiness index (14-period) on 1d timeframe
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    # Where TR = max(H-L, abs(H-Cprev), abs(L-Cprev))
    tr1 = pd.Series(high_1d) - pd.Series(low_1d)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop_raw = 100 * np.log10(tr_sum_14 / (atr_14 * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 1.3 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R3 or R4 (take profit) OR choppiness too low (trending market weakening)
            if close[i] >= r3_aligned[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 or S4 (take profit) OR choppiness too low
            if close[i] <= s3_aligned[i] or chop_aligned[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            if not volume_confirmed[i]:
                signals[i] = 0.0
                continue
                
            # Choppiness regime filter
            chop_value = chop_aligned[i]
            
            # Long conditions: price tests S1 support with volume in ranging market OR breaks above R1 in trending market
            long_condition = False
            if chop_value > 61.8:  # Ranging market - mean reversion at support
                if low[i] <= s1_aligned[i] and close[i] > s1_aligned[i]:
                    long_condition = True
            else:  # Trending market - breakout above resistance
                if close[i] > r1_aligned[i]:
                    long_condition = True
            
            # Short conditions: price tests R1 resistance with volume in ranging market OR breaks below S1 in trending market
            short_condition = False
            if chop_value > 61.8:  # Ranging market - mean reversion at resistance
                if high[i] >= r1_aligned[i] and close[i] < r1_aligned[i]:
                    short_condition = True
            else:  # Trending market - breakdown below support
                if close[i] < s1_aligned[i]:
                    short_condition = True
            
            if long_condition:
                position = 1
                signals[i] = 0.25
            elif short_condition:
                position = -1
                signals[i] = -0.25
    
    return signals