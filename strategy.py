#!/usr/bin/env python3
"""
4h Camarilla Pivot Range Breakout + Volume Spike + Daily EMA Filter
Based on Camarilla pivot levels from daily timeframe. Long when price breaks above R1 with volume
spike and price above daily EMA50. Short when price breaks below S1 with volume spike and price
below daily EMA50. Uses daily EMA50 as trend filter to avoid counter-trend trades.
Designed for low trade frequency with clear breakout logic.
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
    
    # Get daily data for Camarilla pivots and EMA50 (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla pivot levels from previous daily bar
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Range = H - L
    range_1d = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    # R4 = C + ((H-L) * 1.5000)
    # R3 = C + ((H-L) * 1.2500)
    # R2 = C + ((H-L) * 1.1666)
    # R1 = C + ((H-L) * 1.0833)
    # PP = (H + L + C) / 3
    # S1 = C - ((H-L) * 1.0833)
    # S2 = C - ((H-L) * 1.1666)
    # S3 = C - ((H-L) * 1.2500)
    # S4 = C - ((H-L) * 1.5000)
    
    r1 = typical_price + (range_1d * 1.0833)
    s1 = typical_price - (range_1d * 1.0833)
    pp = typical_price  # Pivot point
    
    # Align Camarilla levels to lower timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        above_ema = price > ema_50_1d_aligned[i]
        below_ema = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1, above daily EMA, volume spike
            if (price > r1_aligned[i] and above_ema and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below daily EMA, volume spike
            elif (price < s1_aligned[i] and below_ema and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below pivot point or below daily EMA
            if price < pp_aligned[i] or below_ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above pivot point or above daily EMA
            if price > pp_aligned[i] or above_ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_RangeBreakout_VolumeSpike_DailyEMA50"
timeframe = "4h"
leverage = 1.0