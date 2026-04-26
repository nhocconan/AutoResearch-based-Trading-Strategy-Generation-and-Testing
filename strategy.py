#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike, and choppiness regime filter.
Enters long when price breaks above R1 with bullish 1d trend, volume spike, and choppy market (CHOP > 61.8).
Enters short when price breaks below S1 with bearish 1d trend, volume spike, and choppy market.
Exits when price reverses to opposite Camarilla level or trend changes.
Uses 12h primary timeframe to target 12-37 trades/year (50-150 total over 4 years).
Chop filter reduces whipsaws in strong trends, improving performance in both bull and bear markets.
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
    
    # Get 1d data for trend and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate previous 1d bar's Camarilla pivot levels (R1, S1)
    high_1d_prev = np.roll(df_1d['high'].values, 1)
    low_1d_prev = np.roll(df_1d['low'].values, 1)
    close_1d_prev = np.roll(df_1d['close'].values, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d = high_1d_prev - low_1d_prev
    r1 = pivot + (range_1d * 1.0 / 12.0)
    s1 = pivot - (range_1d * 1.0 / 12.0)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Choppiness Index regime filter (14-period) - calculated on 12h timeframe
    # CHOP > 61.8 = choppy/range market (favor mean reversion/breakouts)
    # CHOP < 38.2 = trending market (favor trend following)
    # We use CHOP > 61.8 to avoid strong trending markets where breakouts fail
    atr_period = 14
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    range_sum = highest_high - lowest_low
    chop = np.zeros_like(close)
    chop[:] = np.nan
    mask = range_sum > 0
    chop[mask] = 100 * np.log10(atr[mask] * np.sqrt(atr_period) / range_sum[mask]) / np.log10(atr_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA, 14 for ATR/CHOP, 1 for pivot)
    start_idx = max(34, 20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 1d bullish trend, volume spike, and choppy market
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and volume_spike[i] and chop[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 1d bearish trend, volume spike, and choppy market
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and volume_spike[i] and chop[i] > 61.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S1 OR 1d trend turns bearish
            if (close[i] < s1_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R1 OR 1d trend turns bullish
            if (close[i] > r1_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeFilter"
timeframe = "12h"
leverage = 1.0