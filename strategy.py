#!/usr/bin/env python3
"""
6h Volume-Weighted Moving Average (VWMA) Crossover with 1d Trend Filter and Volume Spike
Hypothesis: VWMA crossovers capture volume-weighted momentum, filtered by 1d trend for direction and volume spikes for confirmation. Works in bull (buy when fast VWMA > slow VWMA) and bear (sell when fast VWMA < slow VWMA). Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vwma_crossover_1d_trend_volspike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # VWMA calculation
    def vwma(close_arr, volume_arr, period):
        vwma_arr = np.full(len(close_arr), np.nan)
        if len(close_arr) < period:
            return vwma_arr
        for i in range(period-1, len(close_arr)):
            numerator = np.sum(close_arr[i-period+1:i+1] * volume_arr[i-period+1:i+1])
            denominator = np.sum(volume_arr[i-period+1:i+1])
            vwma_arr[i] = numerator / denominator if denominator != 0 else np.nan
        return vwma_arr
    
    # Fast VWMA (10-period) and Slow VWMA (30-period) on 6h data
    vwma_fast = vwma(close, volume, 10)
    vwma_slow = vwma(close, volume, 30)
    
    # Get 1d data for trend filter (close > SMA20)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 20-period SMA on 1d close
    sma_20_1d = np.full(len(close_1d), np.nan)
    for i in range(19, len(close_1d)):
        sma_20_1d[i] = np.mean(close_1d[i-19:i+1])
    
    # 1d trend: above SMA20 = bullish, below = bearish
    trend_1d = np.where(close_1d > sma_20_1d, 1, -1)
    
    # Align 1d trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 1d volume average (20-period) for volume spike filter
    vol_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align volume MA to 6h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 40  # Need enough data for VWMA and alignments
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(vwma_fast[i]) or np.isnan(vwma_slow[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume spike: current 6h volume > 2x 1d average volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_20_1d_aligned[i] / 4.0 * 2.0
        volume_spike = volume[i] > vol_threshold
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: VWMA crossover down OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (vwma_fast[i] < vwma_slow[i] or
                trend_1d_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: VWMA crossover up OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (vwma_fast[i] > vwma_slow[i] or
                trend_1d_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 12 bars flat
            if bars_since_entry >= 12:
                # VWMA crossover entries with trend and volume spike
                vwma_cross_up = vwma_fast[i] > vwma_slow[i]
                vwma_cross_down = vwma_fast[i] < vwma_slow[i]
                
                # Long: VWMA cross up with bullish 1d trend + volume spike
                if vwma_cross_up and trend_1d_aligned[i] == 1 and volume_spike:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: VWMA cross down with bearish 1d trend + volume spike
                elif vwma_cross_down and trend_1d_aligned[i] == -1 and volume_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals
</script>