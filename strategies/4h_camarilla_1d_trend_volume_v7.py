#!/usr/bin/env python3
# 4h_camarilla_1d_trend_volume_v7
# Hypothesis: 4h strategy using 1d Camarilla pivot levels for structure, volume confirmation, and ATR-based trend filter.
# Long: price above daily pivot + volume spike + close > H3 + ATR(14) rising
# Short: price below daily pivot + volume spike + close < L3 + ATR(14) falling
# Uses discrete sizing (±0.30) to minimize fee churn. Target: 75-200 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_1d_trend_volume_v7"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First bar: use same day's data (no look-ahead)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla pivot levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    
    # Resistance levels
    H3 = pivot + (range_ * 1.1 / 4)
    H4 = pivot + (range_ * 1.1 / 2)
    # Support levels
    L3 = pivot - (range_ * 1.1 / 4)
    L4 = pivot - (range_ * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trend filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR slope: rising if current > previous
    atr_rising = atr > np.roll(atr, 1)
    atr_falling = atr < np.roll(atr, 1)
    atr_rising[0] = False
    atr_falling[0] = False
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(H4_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (trend reversal)
            if close[i] < L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 (trend reversal)
            if close[i] > H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Need volume confirmation and ATR trend filter
            volume_confirmed = volume[i] > 2.0 * volume_ma[i]
            
            if volume_confirmed:
                # Long: price above pivot + close > H3 + ATR rising
                if close[i] > pivot_aligned[i] and close[i] > H3_aligned[i] and atr_rising[i]:
                    position = 1
                    signals[i] = 0.30
                # Short: price below pivot + close < L3 + ATR falling
                elif close[i] < pivot_aligned[i] and close[i] < L3_aligned[i] and atr_falling[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals