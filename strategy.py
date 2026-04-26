#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeChop_v1
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA34 trend filter, volume spike (>1.5x median), and choppiness regime filter (CHOP > 50 for mean reversion avoidance). Targets institutional pivot levels with trend alignment and volume conviction while avoiding strong trends where pivots fail. Designed for BTC/ETH to work in ranging markets with occasional breakouts. Uses discrete sizing (0.25) to minimize fee churn. Targets 20-50 trades/year via tight confluence of 4 conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d True Range for Choppiness Index
    tr1 = np.abs(df_1d['high'].values - df_1d['low'].values)
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Choppiness Index: CHOP = 100 * LOG10(SUM(ATR14) / (MAXHIGH - MINLOW)) / LOG10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    chop_raw = 100 * np.log10(sum_atr_14 / (range_14 + 1e-10)) / np.log10(14)
    chop_raw = np.where(range_14 > 0, chop_raw, 50.0)  # Avoid division by zero
    
    # Get 4h data for Camarilla levels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Camarilla levels from previous 4h bar (HLC of prior 4h)
    cam_high = pd.Series(df_4h['high'].values).shift(1).values
    cam_low = pd.Series(df_4h['low'].values).shift(1).values
    cam_close = pd.Series(df_4h['close'].values).shift(1).values
    
    # Camarilla R1, S1, R2, S2 levels (core breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    R2 = cam_close + (cam_high - cam_low) * 1.1 / 6
    S2 = cam_close - (cam_high - cam_low) * 1.1 / 6
    
    # Volume spike filter: volume > 1.5x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    R2_aligned = align_htf_to_ltf(prices, df_4h, R2)
    S2_aligned = align_htf_to_ltf(prices, df_4h, S2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA(34) 1d, chop, Camarilla (need 2 bars for shift), volume median (20), ATR (14)
    start_idx = max(34, 14, 2, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(R2_aligned[i]) or
            np.isnan(S2_aligned[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        chop_val = chop_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        r2_val = R2_aligned[i]
        s2_val = S2_aligned[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
        # Volume spike filter: only trade in above-average volume environments
        volume_spike = volume_val > 1.5 * vol_median_val
        
        # Choppiness regime filter: avoid strong trends (CHOP < 38.2) and extreme chop (CHOP > 61.8)
        # Focus on range-bound markets where pivots work best (38.2 <= CHOP <= 61.8)
        chop_filter = (chop_val >= 38.2) and (chop_val <= 61.8)
        
        if position == 0:
            # Long: break above R1 with volume spike, uptrend, and chop filter
            long_signal = (close_val > r1_val) and \
                          volume_spike and \
                          uptrend and \
                          chop_filter
            
            # Short: break below S1 with volume spike, downtrend, and chop filter
            short_signal = (close_val < s1_val) and \
                           volume_spike and \
                           downtrend and \
                           chop_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # Exit conditions: break below S2 (mean reversion) or ATR trailing stop
            if close_val < s2_val or close_val < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # Exit conditions: break above R2 (mean reversion) or ATR trailing stop
            if close_val > r2_val or close_val > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeChop_v1"
timeframe = "4h"
leverage = 1.0