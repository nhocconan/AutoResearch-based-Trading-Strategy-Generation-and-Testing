#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_EMA200
Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA200 trend filter, volume confirmation, and choppiness regime filter.
Targets 75-200 total trades over 4 years (19-50/year) by using tight entry conditions.
Works in bull/bear markets: EMA200 defines trend regime, volume spike confirms breakout strength,
and choppiness filter avoids whipsaws in ranging markets. Discrete sizing (0.25) minimizes fee drag.
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
    
    # Get 1d data for Camarilla and EMA200
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels from previous 1d bar (completed)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA200 trend filter
    ema_200 = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Choppiness regime filter on 1d: CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    # Calculate CHOP(14) on 1d data
    atr_1d = pd.Series(np.sqrt((df_1d['high'].values - df_1d['low'].values)**2)).rolling(window=14, min_periods=14).mean().values
    max_h = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_l = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_1d * 14 / np.log((max_h - min_l))) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need 1d shift, EMA200, vol avg, chop
    start_idx = max(30, 200, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_200_aligned[i]
        vol_spike = volume_spike[i]
        chop_val = chop_aligned[i]
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with EMA alignment, volume spike, and trending regime
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_spike and 
                            trending_regime)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_spike and 
                             trending_regime)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below EMA200 (trend reversal) OR chop becomes too high (ranging market)
            if close_val < ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA200 (trend reversal) OR chop becomes too high (ranging market)
            if close_val > ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_EMA200"
timeframe = "4h"
leverage = 1.0