#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme Reversion with Volume Spike and Regime Filter.
Long when Williams %R crosses above -80 (oversold exit) with volume > 1.8x average in choppy/transition market (CHOP > 50).
Short when Williams %R crosses below -20 (overbought exit) with volume > 1.8x average in choppy/transition market.
Exit when Williams %R returns to -50 (mean reversion midpoint) or chop regime ends (trending: CHOP < 38.2 or CHOP > 61.8).
Uses 1d for Williams %R calculation (more stable extremes) and 1d for chop filter.
Target: 80-160 total trades over 4 years (20-40/year). Uses Williams %R for mean reversion edge in ranging markets and volume confirmation to avoid false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        williams_r = np.full_like(close, -50.0)  # default to neutral
        
        for i in range(period, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
            if highest_high[i] != lowest_low[i]:
                williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
            else:
                williams_r[i] = -50.0  # avoid division by zero
        return williams_r
    
    williams_r_1d = calculate_williams_r(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d Choppiness Index (CHOP)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(close)
        
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's ATR
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
    
    # Align 1d indicators to 4h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate volume spike (current volume > 1.8x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop_1d_aligned[i]
        williams_r = williams_r_1d_aligned[i]
        
        # Regime filters: avoid extreme trending or ranging conditions
        # Chop regime: 38.2 <= CHOP <= 61.8 = transition/fair value (good for mean reversion)
        is_fair_value = (chop_val >= 38.2) and (chop_val <= 61.8)
        # Avoid strong trends: CHOP < 38.2 = trending up, CHOP > 61.8 = ranging/choppy
        avoid_trending_up = chop_val < 38.2
        avoid_strong_chop = chop_val > 61.8
        
        if position == 0:
            # Long: Williams %R crosses above -80 (exiting oversold) with volume spike in fair value
            if williams_r > -80 and williams_r_1d_aligned[i-1] <= -80 and vol_spike and is_fair_value:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (exiting overbought) with volume spike in fair value
            elif williams_r < -20 and williams_r_1d_aligned[i-1] >= -20 and vol_spike and is_fair_value:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean) or regime deteriorates
            if williams_r >= -50 or avoid_trending_up or avoid_strong_chop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean) or regime deteriorates
            if williams_r <= -50 or avoid_trending_up or avoid_strong_chop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_ExtremeReversion_VolumeSpike_FairValue"
timeframe = "4h"
leverage = 1.0