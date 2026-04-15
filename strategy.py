#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with volume spike and 12h choppiness regime filter
# Long when price breaks above Camarilla R1 (1d) + volume > 2x 20-period avg + 12h chop > 61.8 (range)
# Short when price breaks below Camarilla S1 (1d) + volume > 2x 20-period avg + 12h chop > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (20-40/year).
# Camarilla levels provide high-probability reversal/breakout points. Volume spike confirms participation.
# Choppiness filter ensures we only trade in ranging markets where mean reversion at pivot levels works.
# Works in bull markets (buying dips to S1 in range) and bear markets (selling rallies to R1 in range).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 12h HTF data once before loop for choppiness filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1 = pp + (range_1d * 1.1 / 12)
    s1 = pp - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h Indicator: Choppiness Index (CHOP) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range (TR)
    high_12h_shift = np.roll(high_12h, 1)
    low_12h_shift = np.roll(low_12h, 1)
    close_12h_shift = np.roll(close_12h, 1)
    
    # Set first values to avoid NaN from roll
    high_12h_shift[0] = high_12h[0]
    low_12h_shift[0] = low_12h[0]
    close_12h_shift[0] = close_12h[0]
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - close_12h_shift)
    tr3 = np.abs(low_12h - close_12h_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14) using Wilder's smoothing
    chop_period = 14
    atr = np.zeros_like(tr)
    atr[chop_period-1] = np.mean(tr[:chop_period])
    for i in range(chop_period, len(tr)):
        atr[i] = (atr[i-1] * (chop_period-1) + tr[i]) / chop_period
    
    # Calculate maximum and minimum true range over chop_period
    max_tr = np.zeros_like(tr)
    min_tr = np.zeros_like(tr)
    
    for i in range(chop_period-1, len(tr)):
        max_tr[i] = np.max(tr[i-chop_period+1:i+1])
        min_tr[i] = np.min(tr[i-chop_period+1:i+1])
    
    # Avoid division by zero
    denominator = max_tr - min_tr
    chop = np.where(denominator != 0, 
                    -100 * np.log((sum(tr[i-chop_period+1:i+1] for i in range(chop_period-1, len(tr))) / 
                                 (chop_period * denominator)) / np.log(2)), 100)
    
    # Fix: Calculate CHOP properly using rolling sum
    chop = np.full_like(tr, np.nan)
    tr_sum = np.zeros_like(tr)
    tr_sum[chop_period-1] = np.sum(tr[:chop_period])
    for i in range(chop_period, len(tr)):
        tr_sum[i] = tr_sum[i-1] - tr[i-chop_period] + tr[i]
    
    for i in range(chop_period-1, len(tr)):
        if denominator[i] != 0:
            chop[i] = 100 * np.log10(tr_sum[i] / denominator[i]) / np.log10(chop_period)
        else:
            chop[i] = 50.0  # neutral value when no range
    
    # Align choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # === 4h Indicator: Volume SMA for confirmation ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, chop_period) + 20  # 1d data + 12h CHOP(14) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 (1d)
        # 2. Volume confirmation
        # 3. Choppiness > 61.8 (ranging market)
        if (close[i] > r1_aligned[i]) and \
           vol_confirm and \
           (chop_aligned[i] > 61.8):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 (1d)
        # 2. Volume confirmation
        # 3. Choppiness > 61.8 (ranging market)
        elif (close[i] < s1_aligned[i]) and \
             vol_confirm and \
             (chop_aligned[i] > 61.8):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_CamarillaR1S1_VolumeSpike_12hChop618_v1"
timeframe = "4h"
leverage = 1.0