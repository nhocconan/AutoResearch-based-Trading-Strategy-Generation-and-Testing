#!/usr/bin/env python3
"""
4h_Vortex_Trend_With_Volume_Spike
Hypothesis: The Vortex Indicator captures trend direction with clear crossovers.
In bull markets, VI+ > VI- indicates strength; in bear markets, VI- > VI+ indicates weakness.
Combined with volume spikes and a 1-day trend filter, this reduces false signals.
Targets 25-40 trades/year to minimize fee drag while maintaining edge in both bull/bear regimes.
"""

name = "4h_Vortex_Trend_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for 1-day trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Vortex Indicator (period=14)
    tr = np.maximum(np.abs(high[1:] - low[1:]), 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    vm_plus = np.abs(high - low[:-1])  # |high - prior low|
    vm_minus = np.abs(low - high[:-1]) # |low - prior high|
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    # Sum over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus_sum / tr_sum
    vi_minus = vm_minus_sum / tr_sum
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        if position == 0:
            # LONG: VI+ > VI- (bullish trend) with volume spike and above 1-day EMA34
            if (vi_plus[i] > vi_minus[i] and 
                volume_spike[i] and 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (bearish trend) with volume spike and below 1-day EMA34
            elif (vi_minus[i] > vi_plus[i] and 
                  volume_spike[i] and 
                  close[i] < trend_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- > VI+ or price drops below 1-day EMA34
            if (vi_minus[i] > vi_plus[i] or 
                close[i] < trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ > VI- or price rises above 1-day EMA34
            if (vi_plus[i] > vi_minus[i] or 
                close[i] > trend_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals