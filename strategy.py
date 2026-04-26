#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter_v2
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike, and chop regime filter.
Enters long on break above R1 when 1d trend up (close>EMA34), volume spike, and chop<61.8 (trending).
Enters short on break below S1 when 1d trend down (close<EMA34), volume spike, and chop<61.8.
Uses 12h primary timeframe to target 12-37 trades/year (50-150 total over 4 years).
Chop filter avoids range markets, volume spike confirms breakout strength, 1d EMA34 ensures trend alignment.
Works in bull/bear markets by only trading with 1d trend and requiring volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators (trend and chop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d Choppiness Index (CHOP) for regime filter
    atr_period = 14
    tr1 = pd.Series(df_1d['high']).shift(1) - pd.Series(df_1d['low']).shift(1)
    tr2 = pd.Series(df_1d['high']) - pd.Series(df_1d['close']).shift(1)
    tr3 = pd.Series(df_1d['low']) - pd.Series(df_1d['close']).shift(1)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    max_high = pd.Series(df_1d['high']).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(df_1d['low']).rolling(window=atr_period, min_periods=atr_period).min().values
    chop = 100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(atr_period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # 12h Camarilla levels (self-referential, no alignment needed)
    r1_12h = np.zeros(n)
    s1_12h = np.zeros(n)
    for i in range(n):
        if i < 1:
            r1_12h[i] = 0
            s1_12h[i] = 0
        else:
            r1, s1 = calculate_camarilla(high[i-1], low[i-1], close[i-1])
            r1_12h[i] = r1
            s1_12h[i] = s1
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA34, 20 for volume MA, 14 for chop)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: break above R1, 1d trend up, volume spike, chop<61.8 (trending)
            if (close[i] > r1_12h[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i] and 
                chop_aligned[i] < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: break below S1, 1d trend down, volume spike, chop<61.8 (trending)
            elif (close[i] < s1_12h[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i] and 
                  chop_aligned[i] < 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close below S1 OR 1d trend down OR chop>61.8 (range)
            if (close[i] < s1_12h[i] or 
                close[i] < ema34_1d_aligned[i] or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close above R1 OR 1d trend up OR chop>61.8 (range)
            if (close[i] > r1_12h[i] or 
                close[i] > ema34_1d_aligned[i] or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_RegimeFilter_v2"
timeframe = "12h"
leverage = 1.0