#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_ChopFilter
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA34 trend filter, volume confirmation, and chop regime filter.
Uses fixed position sizing (0.25) to reduce fee churn. Designed for 4h timeframe targeting 100-180 trades over 4 years.
Works in bull/bear markets: In trending regimes (price > EMA34 for longs, < EMA34 for shorts) AND low chop (<61.8),
breakouts at R1/S1 with volume spike capture momentum. Exit on trend reversal or range re-entry.
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
    
    # Get 1d data for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels from previous 1d bar (completed)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    r3 = prev_close + (rng * 1.1 / 4)
    s3 = prev_close - (rng * 1.1 / 4)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Chop regime filter: CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop formula: 100 * log10(sum(TR14)/(n*(HH-LL))) / log10(n)
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * (np.log10(sum_tr) - np.log10(14 * (hh - ll))) / np.log10(14)
    chop_regime = chop < 61.8  # True when trending (chop < 61.8), False when ranging
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Reduced size to lower drawdown and fee churn
    
    # Warmup: need 1d shift, EMA34, vol avg, chop
    start_idx = max(30, 34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        chop_ok = chop_regime[i]
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with EMA alignment, volume spike, and trending regime
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_spike and
                            chop_ok)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_spike and
                             chop_ok)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price re-enters Camarilla range (below S1) OR loses EMA alignment
            # Stronger exit: break S3 or close below EMA
            if close_val < s3_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price re-enters Camarilla range (above R1) OR loses EMA alignment
            # Stronger exit: break R3 or close above EMA
            if close_val > r3_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0