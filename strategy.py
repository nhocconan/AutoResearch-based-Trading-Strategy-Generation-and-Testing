#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_MeanReversion_with_Breakout_Filter
Hypothesis: In 12h timeframe, price tends to mean-revert from daily Camarilla R1/S1 levels with volume confirmation, but only when intraday volatility is low (ATR-based filter). Breakouts at R4/S4 are ignored to avoid false signals in ranging markets. This reduces trade frequency and improves selectivity, targeting 15-35 trades/year. Works in both bull and bear markets by fading extremes rather than chasing momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load daily data once for Camarilla pivot points
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily Camarilla pivot levels R1 and S1 only
    P = (high_daily + low_daily + close_daily) / 3.0
    range_daily = high_daily - low_daily
    r1_daily = P + (range_daily * 0.382)
    s1_daily = P - (range_daily * 0.382)
    
    # Align daily R1/S1 to 12h timeframe
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volatility filter: use 12-period ATR to avoid choppy markets
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(close)
    for i in range(len(close)):
        if i < 12:
            atr[i] = np.mean(tr[:i+1]) if i > 0 else tr[0]
        else:
            atr[i] = np.mean(tr[i-12:i])
    # Normalize ATR by price to get % volatility
    vol_percent = atr / close
    # Only trade when volatility is below median (avoid high-chop periods)
    vol_median = np.percentile(vol_percent[~np.isnan(vol_percent)], 50) if np.sum(~np.isnan(vol_percent)) > 0 else 0.02
    low_vol_filter = vol_percent <= vol_median
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 20:
            volume_avg[i] = np.mean(volume[i-20:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.3 * volume_avg)
    
    # Combined filter: low volatility AND volume confirmation
    filter_ok = low_vol_filter & volume_filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_daily_aligned[i]) or np.isnan(s1_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = r1_daily_aligned[i]
        s1 = s1_daily_aligned[i]
        filt = filter_ok[i]
        
        if position == 0:
            # Long: price rejects S1 with volume and low volatility
            if price > s1 and price < (s1 + (r1 - s1) * 0.25) and filt:
                # Require bullish close (close > midpoint)
                if close[i] > (high[i] + low[i]) / 2:
                    signals[i] = 0.25
                    position = 1
            # Short: price rejects R1 with volume and low volatility
            elif price < r1 and price > (r1 - (r1 - s1) * 0.25) and filt:
                # Require bearish close (close < midpoint)
                if close[i] < (high[i] + low[i]) / 2:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to midpoint or breaks above R1 (failure)
            if price < (r1 + s1) / 2 or price > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint or breaks below S1 (failure)
            if price > (r1 + s1) / 2 or price < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1S1_MeanReversion_with_Breakout_Filter"
timeframe = "12h"
leverage = 1.0