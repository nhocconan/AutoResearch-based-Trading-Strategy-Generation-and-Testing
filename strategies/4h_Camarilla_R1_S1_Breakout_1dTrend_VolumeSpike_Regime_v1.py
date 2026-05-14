#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v1
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation. 
Only long when price > EMA34(1d), short when price < EMA34(1d). Uses ATR-based position sizing (0.25-0.35) 
and includes a choppiness regime filter to avoid whipsaws in sideways markets. 
Designed for 75-150 total trades over 4 years (19-38/year) with strong performance in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for Camarilla calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R1, S1, PP
    camarilla_multiplier = 1.1 / 4
    r1 = close_1d + range_1d * camarilla_multiplier
    pp = typical_price
    s1 = close_1d - range_1d * camarilla_multiplier
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 1d data for EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR for volatility-based position sizing and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # Choppiness regime filter: avoid trading in high chop (range-bound) markets
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We only trade when CHOP < 61.8 (not strongly ranging)
    hl_range = np.maximum(high, low) - np.minimum(high, low)
    sum_range = pd.Series(hl_range).rolling(window=14, min_periods=14).sum()
    abs_close_diff = np.abs(np.diff(close, prepend=close[0]))
    sum_abs_diff = pd.Series(abs_close_diff).rolling(window=14, min_periods=14).sum()
    chop = 100 * np.log10(sum_range / (sum_abs_diff + 1e-10)) / np.log10(14)
    chop_filter = chop < 61.8  # Avoid strongly ranging markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Skip if in strongly ranging market (high chop)
        if not chop_filter[i]:
            # Flatten position in choppy markets
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # ATR-based position sizing (0.25-0.35)
        atr_ratio = atr[i] / (np.mean(atr[max(0, i-50):i+1]) + 1e-10)
        if atr_ratio > 1.3:  # High volatility
            base_size = 0.35
        elif atr_ratio < 0.7:  # Low volatility
            base_size = 0.25
        else:  # Normal volatility
            base_size = 0.30
        
        # Long logic: price breaks above R1 with volume spike and above 1d EMA34
        if close[i] > r1_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S1 with volume spike and below 1d EMA34
        elif close[i] < s1_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to pivot point or opposite breakout
        elif position == 1 and (close[i] < pp_aligned[i] or close[i] < s1_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > pp_aligned[i] or close[i] > r1_aligned[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0