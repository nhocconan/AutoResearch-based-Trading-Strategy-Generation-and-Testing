#!/usr/bin/env python3
"""
6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v3
Hypothesis: Use Camarilla pivot levels from 1d timeframe to identify breakout/breakdown zones.
Go long when price breaks above R1 with volume > 1.5x 24-period average and ATR(14) > 0.5 * ATR(50) (volatility filter).
Go short when price breaks below S1 with same conditions.
Exit when price retests the opposite level (S1 for longs, R1 for shorts) or when volatility drops.
Uses 1d Camarilla levels for stability and 6f for execution. Designed to work in both bull (breakouts) and bear (breakdowns) markets with volatility filter to avoid chop.
Targets ~15-25 trades/year via strict breakout conditions + volume + volatility confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day (using typical formula)
    # R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), R2 = C + ((H-L) * 1.1/6), R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12), S2 = C - ((H-L) * 1.1/6), S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 1 and not (np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1])):
            hl_range = high_1d[i-1] - low_1d[i-1]
            camarilla_r1[i] = close_1d[i-1] + (hl_range * 1.1 / 12)
            camarilla_s1[i] = close_1d[i-1] - (hl_range * 1.1 / 12)
        else:
            camarilla_r1[i] = np.nan
            camarilla_s1[i] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # Volatility filter: ATR(14) > 0.5 * ATR(50) to avoid low volatility chop
    def calculate_atr(high, low, close, period):
        atr = np.full_like(close, np.nan)
        if len(close) < period:
            return atr
        tr = np.zeros_like(close)
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        # Wilder smoothing
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14 = calculate_atr(high, low, close, 14)
    atr_50 = calculate_atr(high, low, close, 50)
    vol_filter = (atr_14 > 0) & (atr_50 > 0) & (atr_14 > 0.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 50)  # Need enough data for vol MA and ATR50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        # Volume and volatility confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        vol_cond = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume + volatility
            if close[i] > camarilla_r1_aligned[i] and vol_confirm and vol_cond:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + volatility
            elif close[i] < camarilla_s1_aligned[i] and vol_confirm and vol_cond:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price retests S1 or volatility drops
            if close[i] < camarilla_s1_aligned[i] or not vol_cond:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price retests R1 or volatility drops
            if close[i] > camarilla_r1_aligned[i] or not vol_cond:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_v3"
timeframe = "6h"
leverage = 1.0