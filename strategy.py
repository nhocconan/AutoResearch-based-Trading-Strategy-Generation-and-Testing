#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_Regime_v1
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike (>2.0x average), and choppiness regime filter (CHOP > 61.8 = range) captures reliable breakouts in both bull and bear markets. Uses discrete sizing (0.30) and close-based exits. Designed for 4h timeframe to target 20-50 trades/year, minimizing fee drag while maintaining edge. The regime filter avoids trending markets where breakouts fail, focusing on range-bound conditions where mean reversion at pivots works best.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR(14) for choppiness calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 4h (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Average volume for confirmation (24-period SMA = 4h * 6 = 1 day)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Choppiness Index: CHOP > 61.8 = ranging market (good for mean reversion at pivots)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high) - min(low))) / log10(14)
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_high14 - min_low14 + 1e-10)) / np.log10(14)
    chop_filter = chop > 61.8  # Only trade in ranging markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.30
    
    # Warmup: max of EMA(34), volume(24), ATR(14), CHOP(14)
    start_idx = max(34, 24, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        chop_val = chop[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(chop_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        regime_ok = chop_val > 61.8
        
        # Long: price CLOSES above R1 with 1d uptrend, volume, and ranging regime
        long_condition = (close_val > r1_val) and (close_val > ema_val) and volume_confirmed and regime_ok
        # Short: price CLOSES below S1 with 1d downtrend, volume, and ranging regime
        short_condition = (close_val < s1_val) and (close_val < ema_val) and volume_confirmed and regime_ok
        
        # Exit: price retests broken level
        long_exit = (position == 1 and close_val <= r1_val)
        short_exit = (position == -1 and close_val >= s1_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike_Regime_v1"
timeframe = "4h"
leverage = 1.0