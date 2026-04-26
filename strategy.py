#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_Regime_v1
Hypothesis: Trade 12h Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume confirmation.
Uses choppiness regime filter to avoid whipsaws in ranging markets. R1/S1 levels provide
more frequent but reliable breakouts than R3/S3 when combined with trend and volume.
In bull markets: price breaks above R1 with 1d uptrend → long.
In bear markets: price breaks below S1 with 1d downtrend → short.
Volume confirmation ensures breakouts have participation.
Choppiness filter avoids false signals in low-volatility ranging periods.
Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    # Volume confirmation: current volume > 2.0 * 20-period average (approx 10d average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # Choppiness regime filter: avoid ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) / (max(high,n)-min(low,n)))
    # Simplified: use rolling max/min range vs ATR sum
    lookback = 14
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr1 = np.concatenate([[0], tr1])  # align length
    atr_sum = pd.Series(tr1).rolling(window=lookback, min_periods=lookback).sum().values
    max_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    min_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    chop_denom = max_high - min_low
    chop_denom = np.where(chop_denom == 0, 1, chop_denom)  # avoid div by zero
    chopiness = 100 * np.log10(atr_sum / chop_denom) / np.log10(lookback)
    # Market is trending when CHOP < 38.2, ranging when CHOP > 61.8
    # We want trending markets: CHOP < 50 (middle ground for more signals)
    chop_filter = chopiness < 50.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of 1d EMA(34), volume MA(20), chop lookback(14), and need 1d data
    start_idx = max(34, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(chop_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        chop_ok = chop_filter[i]
        trend_up = close_val > ema_34_1d_aligned[i]   # 1d uptrend
        trend_down = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirm AND chop OK AND 1d uptrend
            long_signal = (close_val > r1_aligned[i]) and vol_conf and chop_ok and trend_up
            
            # Short: price breaks below S1 AND volume confirm AND chop OK AND 1d downtrend
            short_signal = (close_val < s1_aligned[i]) and vol_conf and chop_ok and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price drops below S1 (failed breakout) OR 1d trend flips down OR chop too high
            if (close_val < s1_aligned[i]) or (not trend_up) or (not chop_ok):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 (failed breakdown) OR 1d trend flips up OR chop too high
            if (close_val > r1_aligned[i]) or (not trend_down) or (not chop_ok):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Regime_v1"
timeframe = "12h"
leverage = 1.0