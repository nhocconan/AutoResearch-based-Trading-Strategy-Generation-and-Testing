#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: 4h strategy using Camarilla R1/S1 breakouts from previous 1d bar with 1d EMA50 trend filter, volume confirmation (>2.0x 20-period average), and choppiness regime filter (CHOP(14) < 61.8 to avoid whipsaw in ranging markets). R1/S1 levels provide more frequent but reliable breakouts when aligned with daily trend and volume. Designed for BTC/ETH robustness in both bull and bear markets via trend filter and regime avoidance. Targets 75-200 trades over 4 years (19-50/year) with 0.25 position size. Uses discrete levels to minimize fee drag.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Get 1d data for Camarilla R1/S1 levels (from previous completed 1d bar)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.0833)   # R1 level
    s1 = prev_close - (rng * 1.0833)   # S1 level
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Choppiness regime filter: CHOP(14) < 61.8 to avoid ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / np.log10(highest_high - lowest_low + 1e-10) / np.log10(14))
    chop[np.isnan(chop) | (chop < 0) | (chop > 100)] = 50  # clamp invalid values
    chop_filter = chop < 61.8  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need 1d EMA50 (50), 1d shift(1) for Camarilla, vol avg (20), chop (14+13)
    start_idx = max(50 + 1, 1 + 1, 20, 14 + 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with 1d EMA50 alignment, volume confirmation, and trending regime
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_conf and 
                            chop_ok)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_conf and 
                             chop_ok)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA50 (trend reversal)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 1d EMA50 (trend reversal)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0