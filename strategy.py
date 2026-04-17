#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme Reversion with Volume Spike and Chop Regime Filter.
Long when Williams %R < -80 (oversold) with volume > 1.8x average in choppy market (CHOP > 61.8).
Short when Williams %R > -20 (overbought) with volume > 1.8x average in choppy market.
Exit when Williams %R reverts to -50 level or chop regime ends (CHOP < 38.2).
Uses 12h for Williams %R and volume, 1d for chop filter.
Target: 50-150 total trades over 4 years (12-37/year). Uses tighter Williams %R thresholds and volume confirmation to reduce trade frequency.
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
    
    # Get 1d data for chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Williams %R (14-period)
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        wr = np.full_like(close, -50.0)  # default neutral
        for i in range(period-1, len(close)):
            if highest_high[i] != lowest_low[i]:
                wr[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
            else:
                wr[i] = -50
        return wr
    
    wr_12h = calculate_williams_r(high, low, close, 14)
    
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
    
    # Align 1d chop to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate volume spike (current volume > 1.8x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(wr_12h[i]) or np.isnan(chop_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop_1d_aligned[i]
        wr = wr_12h[i]
        
        # Chop regime: CHOP > 61.8 = ranging (good for mean reversion at extremes)
        is_choppy = chop_val > 61.8
        # Exit chop regime: CHOP < 38.2 = trending (avoid false signals)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with volume spike in choppy market
            if wr < -80 and vol_spike and is_choppy:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with volume spike in choppy market
            elif wr > -20 and vol_spike and is_choppy:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R reverts to -50 OR chop regime ends (trending)
            if wr >= -50 or is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R reverts to -50 OR chop regime ends (trending)
            if wr <= -50 or is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_ExtremeReversion_VolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0