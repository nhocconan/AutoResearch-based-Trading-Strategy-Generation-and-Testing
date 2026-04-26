#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume spike, and choppiness regime filter.
R1/S1 levels provide frequent but reliable breakouts. 1d EMA34 trend filter ensures alignment with daily momentum.
Volume spike confirms institutional participation. Choppiness regime filter (CHOP > 61.8) avoids whipsaws in ranging markets.
Designed for 12h timeframe to target 50-150 trades over 4 years (12-37/year). Works in both bull and bear markets by
combining trend following with volatility-based regime filtering to reduce false signals.
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
    
    # Get 1d data for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate choppiness regime filter: CHOP(14) > 61.8 = ranging (avoid)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (max(high) - min(low)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = highest_high - lowest_low
    chop = 100 * np.log10(sum_atr_14 / np.log10(14) / range_14)
    chop = np.where(range_14 == 0, 100, chop)  # avoid division by zero
    chop_regime = chop > 61.8  # True when choppy/ranging (we avoid entries)
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_1d['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_1d['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_1d['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)
    s1 = pivot - (range_hl * 1.1 / 12.0)
    
    # Align Camarilla levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA(34), volume MA, ATR, chop
    start_idx = max(34, 20, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        close_val = close[i]
        trend_1d_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_1d_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        vol_spike = volume_spike[i]
        in_chop = chop_regime[i]  # True when choppy (avoid entries)
        
        if position == 0:
            # Long: price breaks above R1 AND 1d trend up AND volume spike AND NOT choppy
            long_signal = (close_val > r1_aligned[i]) and trend_1d_up and vol_spike and (not in_chop)
            
            # Short: price breaks below S1 AND 1d trend down AND volume spike AND NOT choppy
            short_signal = (close_val < s1_aligned[i]) and trend_1d_down and vol_spike and (not in_chop)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: trend flips down OR price hits ATR stoploss OR chop regime enters
            if (not trend_1d_up) or (close_val < entry_price - 2.0 * atr[i]) or in_chop:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: trend flips up OR price hits ATR stoploss OR chop regime enters
            if (not trend_1d_down) or (close_val > entry_price + 2.0 * atr[i]) or in_chop:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0