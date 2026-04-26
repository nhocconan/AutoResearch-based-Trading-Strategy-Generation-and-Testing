#!/usr/bin/env python3
"""
4h_Camarilla_PP_Bounce_1dEMA50_Trend_VolumeFilter
Hypothesis: In ranging markets (CHOP > 61.8), price tends to revert to the Camarilla pivot point (PP) from extremes.
Long when: price crosses above PP from below + 1d EMA50 uptrend + volume > 1.5 * avg volume.
Short when: price crosses below PP from above + 1d EMA50 downtrend + volume > 1.5 * avg volume.
Exit when: price reaches opposite Camarilla level (S1 for long, R1 for short) or reverts back to PP.
Uses discrete 0.25 position size. Targets ~30 trades/year to avoid fee drag.
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
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R1, S1, PP (pivot point)
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 4h timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_avg)
    
    # Choppiness Index (CHOP) regime filter - using 14-period
    # CHOP > 61.8 = ranging market (good for mean reversion)
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr * np.sqrt(atr_period) / (max_high - min_low)) / np.log10(atr_period)
    chop = np.where((max_high - min_low) > 0, chop_raw, 50.0)  # default to neutral when range=0
    chop_regime = chop > 61.8  # ranging regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for volume avg, 50 for 1d EMA, 14 for ATR/CHOP
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for PP cross with trend and volume confirmation
            # Long: cross above PP from below + 1d EMA50 uptrend + volume filter + chop > 61.8
            long_entry = (close_val > camarilla_pp_aligned[i]) and (close[i-1] <= camarilla_pp_aligned[i-1]) and \
                       (ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]) and \
                       volume_filter[i] and \
                       chop_regime[i]
            # Short: cross below PP from above + 1d EMA50 downtrend + volume filter + chop > 61.8
            short_entry = (close_val < camarilla_pp_aligned[i]) and (close[i-1] >= camarilla_pp_aligned[i-1]) and \
                        (ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]) and \
                        volume_filter[i] and \
                        chop_regime[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price reaches S1 (opposite extreme) or reverts back to PP
            if (close_val >= camarilla_s1_aligned[i]) or (close_val <= camarilla_pp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reaches R1 (opposite extreme) or reverts back to PP
            if (close_val <= camarilla_r1_aligned[i]) or (close_val >= camarilla_pp_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_PP_Bounce_1dEMA50_Trend_VolumeFilter"
timeframe = "4h"
leverage = 1.0