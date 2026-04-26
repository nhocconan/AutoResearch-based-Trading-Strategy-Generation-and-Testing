#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_v1
Hypothesis: Use TRIX (12) momentum with volume confirmation and choppiness regime filter.
Long when TRIX crosses above zero AND volume spike AND choppy market (CHOP > 61.8).
Short when TRIX crosses below zero AND volume spike AND choppy market.
Chop filter avoids whipsaws in strong trends, captures reversals in ranging markets.
Works in both bull and bear markets by focusing on mean reversion in choppy regimes.
Target: 75-200 total trades over 4 years.
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
    
    # Calculate TRIX (12) - triple exponential moving average
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change() * 100  # Percentage change
    trix_values = trix.values
    
    # Calculate TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_values).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # Choppiness Index (14) - measures if market is choppy (trending vs ranging)
    # CHOP > 61.8 = ranging/choppy market (good for mean reversion)
    # CHOP < 38.2 = trending market (avoid for this strategy)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_sum = highest_high - lowest_low
    range_sum = np.where(range_sum == 0, 1e-10, range_sum)
    
    chop = 100 * np.log10(atr * np.sqrt(atr_period) / range_sum) / np.log10(atr_period)
    chop_values = chop
    
    # Chop regime: True when choppy (CHOP > 61.8)
    chop_regime = chop_values > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of TRIX calculation, volume MA, ATR calculation
    start_idx = max(12*3 + 9, 20, 14) + 5  # Extra buffer for stability
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix_values[i]) or
            np.isnan(trix_signal[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(chop_values[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        trix_val = trix_values[i]
        trix_sig = trix_signal[i]
        vol_conf = volume_confirm[i]
        is_chop = chop_regime[i]
        
        # TRIX crosses above/below signal line
        trix_cross_up = (i > 0 and trix_values[i-1] <= trix_signal[i-1] and trix_val > trix_sig)
        trix_cross_down = (i > 0 and trix_values[i-1] >= trix_signal[i-1] and trix_val < trix_sig)
        
        if position == 0:
            # Long: TRIX bullish cross AND volume confirm AND choppy regime
            long_signal = trix_cross_up and vol_conf and is_chop
            
            # Short: TRIX bearish cross AND volume confirm AND choppy regime
            short_signal = trix_cross_down and vol_conf and is_chop
            
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
            # Exit: TRIX crosses below signal line OR chop regime ends
            if trix_cross_down or not is_chop:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: TRIX crosses above signal line OR chop regime ends
            if trix_cross_up or not is_chop:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0