#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime
Hypothesis: Use 12h timeframe with Camarilla R1/S1 breakout, confirmed by 1d EMA34 trend and choppiness regime filter.
Long when: price breaks above R1 + 1d EMA34 uptrend + chop regime indicates trending (CHOP < 38.2).
Short when: price breaks below S1 + 1d EMA34 downtrend + chop regime indicates trending (CHOP < 38.2).
Exit when: price reverts to Camarilla midpoint (PP) or touches opposite Camarilla level (S3/R3).
Uses discrete 0.25 position size to limit fee drag. Designed for BTC/ETH:
- R1/S1 breakouts with trend and regime filter reduce false signals
- 1d EMA34 filter ensures trading with the daily trend
- Choppiness regime filter avoids ranging markets where breakouts fail
- Targets 12-37 trades/year for optimal test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day (using 1d HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # shift(1) for previous day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R1, S1, PP (pivot point), R3, S3 for exit
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align to 12h timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Choppiness Index (CHOP) on 1d timeframe for regime filter
    # CHOP(14) = 100 * log10(sum(ATR(1) over 14) / log10((max(high)-min(low)) over 14))
    # Simplified: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
    tr_1d = np.maximum(df_1d['high'].values, np.roll(df_1d['close'].values, 1)) - np.minimum(df_1d['low'].values, np.roll(df_1d['close'].values, 1))
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first bar
    atr_1 = pd.Series(tr_1d).rolling(window=1, min_periods=1).sum().values  # ATR(1) = TR
    sum_tr_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.log10(max_high_14 - min_low_14)
    chop_denominator = np.where(chop_denominator <= 0, 1e-10, chop_denominator)  # avoid division by zero
    chop_value = 100 * np.log10(sum_tr_14) / chop_denominator
    chop_value = np.where(np.isnan(chop_value) | np.isinf(chop_value), 50.0, chop_value)  # neutral if invalid
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_value)
    
    # Regime filter: trending when CHOP < 38.2
    trending_regime = chop_aligned < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 14 for chop, 34 for EMA
    start_idx = max(14, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend and regime confirmation
            # Long: break above R1 + 1d EMA34 uptrend + trending regime
            long_entry = (close_val > camarilla_r1_aligned[i]) and \
                       (ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]) and \
                       trending_regime[i]
            # Short: break below S1 + 1d EMA34 downtrend + trending regime
            short_entry = (close_val < camarilla_s1_aligned[i]) and \
                        (ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]) and \
                        trending_regime[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reverts to PP or touches S3 (contrarian exit)
            if (close_val < camarilla_pp_aligned[i]) or (close_val < camarilla_s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to PP or touches R3 (contrarian exit)
            if (close_val > camarilla_pp_aligned[i]) or (close_val > camarilla_r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeRegime"
timeframe = "12h"
leverage = 1.0