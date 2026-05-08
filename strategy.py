#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Vortex Indicator with 1d trend filter and volume confirmation
# Measures trend strength via VI+ and VI- lines. Follows higher timeframe trend
# Requires volume spike to confirm breakouts. Designed for low trade frequency
# Target: 20-50 total trades over 4 years = 5-12/year

name = "4h_Vortex_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Vortex Indicator (period=14) on 4h data
    tr1 = np.maximum(high[1:], low[:-1]) - np.minimum(low[1:], high[:-1])
    tr1 = np.concatenate([[np.nan], tr1])  # align with original arrays
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan
    
    tr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    vi_plus = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values / tr_sum
    vi_minus = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values / tr_sum
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vi_plus[i]) or 
            np.isnan(vi_minus[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        vi_p = vi_plus[i]
        vi_m = vi_minus[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: VI+ > VI- (bullish trend) + uptrend + volume spike
            if (vi_p > vi_m and 
                close[i] > ema34_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: VI- > VI+ (bearish trend) + downtrend + volume spike
            elif (vi_m > vi_p and 
                  close[i] < ema34_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: trend turns bearish OR price breaks below trend
            if (vi_m >= vi_p or close[i] < ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: trend turns bullish OR price breaks above trend
            if (vi_p >= vi_m or close[i] > ema34_1d_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals