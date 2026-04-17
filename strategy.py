#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme Reversion with Volume Spike and Chop Regime Filter.
Long when Williams %R < -80 (oversold) with volume > 2.0x average in choppy market (CHOP > 61.8).
Short when Williams %R > -20 (overbought) with volume > 2.0x average in choppy market.
Exit when Williams %R returns to -50 (mean reversion) or chop regime ends (CHOP < 38.2).
Uses 12h for Williams %R and volume, 1d for chop filter.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R captures extreme reversals in ranging markets,
which works in both bull (buy dips) and bear (sell rallies) regimes when combined with chop filter.
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
    
    # Get 12h data for Williams %R calculation (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for chop filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Over 14 periods
    def calculate_williams_r(high, low, close, period=14):
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        williams_r = np.full_like(close, -50.0)  # default to neutral
        
        for i in range(period, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
            if highest_high[i] != lowest_low[i]:
                williams_r[i] = ((highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])) * -100
            else:
                williams_r[i] = -50  # avoid division by zero
        return williams_r
    
    williams_r_12h = calculate_williams_r(high_12h, low_12h, close_12h, 14)
    
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
    
    # Align indicators to primary timeframe (12h)
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate volume spike (current volume > 2.0x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_12h_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop_1d_aligned[i]
        williams_r = williams_r_12h_aligned[i]
        
        # Chop regime: CHOP > 61.8 = ranging (good for mean reversion at extremes)
        is_choppy = chop_val > 61.8
        # Exit chop regime: CHOP < 38.2 = trending (avoid false signals)
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with volume spike in choppy market
            if williams_r < -80 and vol_spike and is_choppy:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with volume spike in choppy market
            elif williams_r > -20 and vol_spike and is_choppy:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) OR chop regime ends (trending)
            if williams_r >= -50 or is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) OR chop regime ends (trending)
            if williams_r <= -50 or is_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_ExtremeReversion_VolumeSpike_ChopRegime"
timeframe = "12h"
leverage = 1.0