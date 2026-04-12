#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
    # Uses Alligator (SMMA5/8/13) for trend direction and entry timing
    # Volume spike confirms momentum
    # Chop regime filter avoids whipsaws in ranging markets
    # Discrete sizing 0.25 to minimize fee churn. Target: 12-30 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(source, period):
        n = len(source)
        smma = np.full(n, np.nan)
        if n < period:
            return smma
        smma[period-1] = np.mean(source[:period])
        for i in range(period, n):
            smma[i] = (smma[i-1] * (period-1) + source[i]) / period
        return smma
    
    # Williams Alligator: Jaw(13), Teeth(8), Lips(5)
    jaw = smma(close_12h, 13)
    teeth = smma(close_12h, 8)
    lips = smma(close_12h, 5)
    
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Get 1d data for volume and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Volume spike: 1d volume > 2.0 * 20-period average
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Chopiness Index (14-period)
    def chopiness_index(high, low, close, period=14):
        n = len(high)
        chop = np.full(n, np.nan)
        if n < period*2:
            return chop
        # True Range
        tr = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Sum of TR over period
        tr_sum = np.full(n, np.nan)
        for i in range(period, n):
            tr_sum[i] = np.sum(tr[i-period+1:i+1])
        # Highest high and lowest low over period
        hh = np.full(n, np.nan)
        ll = np.full(n, np.nan)
        for i in range(period-1, n):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        # Chop formula
        for i in range(period, n):
            if tr_sum[i] > 0 and hh[i] > ll[i]:
                chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        return chop
    
    chop_1d = chopiness_index(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment: all three lines in order
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Chop regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
        ranging_regime = chop_1d_aligned[i] > 61.8
        trending_regime = chop_1d_aligned[i] < 38.2
        
        # Entry logic
        long_entry = False
        short_entry = False
        
        if trending_regime and bullish_alignment and volume_spike_1d_aligned[i]:
            # Trending bull: go long on Alligator alignment
            long_entry = True
        elif trending_regime and bearish_alignment and volume_spike_1d_aligned[i]:
            # Trending bear: go short on Alligator alignment
            short_entry = True
        elif ranging_regime:
            # Ranging: fade extremes (simplified - could add BB or Donchian here)
            # For now, use Alligator crosses in ranging market
            if lips_aligned[i] > teeth_aligned[i] and lips_aligned[i-1] <= teeth_aligned[i-1]:
                # Lips crossing above teeth in range = potential long
                long_entry = True
            elif lips_aligned[i] < teeth_aligned[i] and lips_aligned[i-1] >= teeth_aligned[i-1]:
                # Lips crossing below teeth in range = potential short
                short_entry = True
        
        # Exit logic: reverse Alligator alignment or chop becomes extreme
        long_exit = bearish_alignment or (chop_1d_aligned[i] > 70)
        short_exit = bullish_alignment or (chop_1d_aligned[i] > 70)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_alligator_chop_volume_v1"
timeframe = "12h"
leverage = 1.0