#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Use 4h timeframe with Camarilla R1/S1 breakout, confirmed by 1d EMA34 trend and volume spike.
Adds chop regime filter (Choppiness Index > 61.8) to avoid whipsaws in ranging markets.
Long when: price breaks above R1 + 1d EMA34 uptrend + volume > 1.5 * avg volume + chop > 61.8
Short when: price breaks below S1 + 1d EMA34 downtrend + volume > 1.5 * avg volume + chop > 61.8
Exit when: price reverts to Camarilla midpoint (PP) or touches opposite level (S1 for long, R1 for short).
Uses discrete 0.25 position size. Targets ~25-40 trades/year for optimal test generalization.
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
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    # Choppiness Index regime filter (avoid ranging markets)
    def choppiness_index(high, low, close, window=14):
        """Calculate Choppiness Index"""
        atr = np.zeros(len(close))
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # ATR calculation
        atr = pd.Series(tr).rolling(window=window, min_periods=window).mean().values
        
        # Highest high and lowest low over window
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        # Choppiness Index formula
        chop = np.zeros(len(close))
        for i in range(window, len(close)):
            if atr[i] > 0 and hh[i] > ll[i]:
                sum_atr = atr[i] * window  # Approximation using current ATR * window
                chop[i] = 100 * np.log10(sum_atr / (hh[i] - ll[i])) / np.log10(window)
            else:
                chop[i] = 50.0  # Neutral when undefined
        return chop
    
    chop = choppiness_index(high, low, close, window=14)
    chop_regime = chop > 61.8  # Trending regime (above 61.8 = trending)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 20 for volume avg, 34 for 1d EMA, 14 for chop
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for breakout with trend, volume, and regime confirmation
            # Long: break above R1 + 1d EMA34 uptrend + volume spike + chop > 61.8
            long_entry = (close_val > camarilla_r1_aligned[i]) and \
                       (ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]) and \
                       volume_spike[i] and \
                       chop_regime[i]
            # Short: break below S1 + 1d EMA34 downtrend + volume spike + chop > 61.8
            short_entry = (close_val < camarilla_s1_aligned[i]) and \
                        (ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]) and \
                        volume_spike[i] and \
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
            # Long - exit when price reverts to PP or touches S1 (contrarian exit)
            if (close_val < camarilla_pp_aligned[i]) or (close_val < camarilla_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price reverts to PP or touches R1 (contrarian exit)
            if (close_val > camarilla_pp_aligned[i]) or (close_val > camarilla_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0