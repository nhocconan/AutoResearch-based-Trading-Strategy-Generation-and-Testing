#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime
Hypothesis: Uses 4h timeframe with Camarilla R1/S1 breakouts filtered by 1d EMA50 trend, volume confirmation, and choppiness regime (CHOP > 61.8 = range, CHOP < 38.2 = trend). Only takes breakouts in trending regimes aligned with 1d EMA50 direction. Designed for BTC/ETH to work in both bull and bear markets by avoiding false breakouts in sideways markets. Targets 20-50 trades/year to minimize fee drag.
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
    
    # Get 1d data for EMA50 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Camarilla levels from previous completed 1d bar
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Choppiness Index (CHOP) on 1d - regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14))
    tr1 = np.maximum(df_1d['high'].values - df_1d['low'].values,
                     np.maximum(np.abs(df_1d['high'].values - df_1d['close'].shift(1).values),
                                np.abs(df_1d['low'].values - df_1d['close'].shift(1).values)))
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    hh14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(sum_atr14) / np.log10(hh14 - ll14) / np.log10(14)
    chop_raw = np.where((hh14 - ll14) > 0, chop_raw, 50.0)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to minimize fee churn
    
    # Warmup: need 1d EMA50 (50), 1d shift(1) for Camarilla, vol avg (20), CHOP (14+14)
    start_idx = max(50 + 24, 1 + 24, 20, 14 + 14 + 24)  # ~204 bars for 1d EMA50 warmup (24 4h bars per day)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1d EMA50 alignment, volume confirmation, and trending regime (CHOP < 38.2)
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_conf and 
                            chop_val < 38.2)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_conf and 
                             chop_val < 38.2)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 (trend reversal) OR choppy regime (CHOP > 61.8)
            if close_val < ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 (trend reversal) OR choppy regime (CHOP > 61.8)
            if close_val > ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0