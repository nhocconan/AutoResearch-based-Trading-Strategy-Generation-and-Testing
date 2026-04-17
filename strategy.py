#!/usr/bin/env python3
"""
Hypothesis: 1h 4h EMA Cross with 1d Volume Spike and Chop Regime Filter.
Long when 4h EMA21 > EMA50 with volume > 2.0x average and choppy market (CHOP > 61.8).
Short when 4h EMA21 < EMA50 with volume > 2.0x average and choppy market.
Exit when EMA cross reverses or chop regime ends (trending).
Uses 4h for EMA trend, 1d for volume and chop filter, 1h for entry timing.
Target: 60-150 total trades over 4 years (15-37/year). Discreet sizing 0.20 to control drawdown.
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
    
    # Get 4h data for EMA trend
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume and chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h EMA21 and EMA50
    def calculate_ema(data, span):
        return pd.Series(data).ewm(span=span, adjust=False, min_periods=span).mean().values
    
    ema21_4h = calculate_ema(close_4h, 21)
    ema50_4h = calculate_ema(close_4h, 50)
    
    # Align 4h EMA to 1h timeframe
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d volume spike (current volume > 2.0x 20-period average)
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (volume_ma_1d * 2.0)
    
    # Calculate 1d Choppiness Index (CHOP)
    def calculate_chop(high, low, close, period=14):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's ATR
        atr = np.zeros_like(close)
        if len(tr) > period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Max true range over period
        max_tr = np.zeros_like(close)
        for i in range(period, len(close)):
            max_tr[i] = np.max(tr[i-period+1:i+1])
        
        # Chop formula: 100 * log10(atr_sum / max_tr) / log10(period)
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if max_tr[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / max_tr[i]) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 1h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema21 = ema21_4h_aligned[i]
        ema50 = ema50_4h_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        chop_val = chop_1d_aligned[i]
        
        # Chop regime: CHOP > 61.8 = ranging (good for trend following in chop)
        is_choppy = chop_val > 61.8
        # Exit chop regime: CHOP < 38.2 = trending (avoid false signals)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long: EMA21 > EMA50 with volume spike in choppy market
            if ema21 > ema50 and vol_spike and is_choppy:
                signals[i] = 0.20
                position = 1
            # Short: EMA21 < EMA50 with volume spike in choppy market
            elif ema21 < ema50 and vol_spike and is_choppy:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: EMA cross reverses OR chop regime ends (trending)
            if ema21 <= ema50 or is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: EMA cross reverses OR chop regime ends (trending)
            if ema21 >= ema50 or is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hEMA_Cross_VolumeSpike_ChopRegime"
timeframe = "1h"
leverage = 1.0