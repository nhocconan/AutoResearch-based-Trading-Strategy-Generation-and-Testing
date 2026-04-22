#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian Channel Breakout with 1-day ATR Filter and Volume Spike.
Long when price breaks above Donchian(20) high with ATR(14) > 1.5x ATR(50) and volume spike.
Short when price breaks below Donchian(20) low with ATR(14) > 1.5x ATR(50) and volume spike.
Exit when price crosses Donchian midline or ATR volatility collapses.
Donchian channels provide clear breakout levels; ATR filter ensures trades occur in volatile regimes;
volume spike confirms institutional interest. Designed for low trade frequency by requiring volatility expansion.
Works in both bull and bear markets by following breakouts in the direction of volatility expansion.
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
    
    # Donchian Channel: 20-period high/low
    def donchian_channel(high, low, period):
        upper = np.full_like(high, np.nan, dtype=float)
        lower = np.full_like(low, np.nan, dtype=float)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    dc_upper, dc_lower = donchian_channel(high, low, 20)
    dc_mid = (dc_upper + dc_lower) / 2.0
    
    # ATR: Average True Range
    def atr(high, low, close, period):
        tr = np.full_like(high, np.nan, dtype=float)
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_vals = np.full_like(tr, np.nan, dtype=float)
        atr_vals[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr_14 = atr(high, low, close, 14)
    atr_50 = atr(high, low, close, 50)
    
    # Load 1d data for ATR filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 14-period ATR on 1d for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_14_1d = atr(high_1d, low_1d, close_1d, 14)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(dc_mid[i]) or
            np.isnan(atr_14[i]) or np.isnan(atr_50[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: short-term ATR > 1.5x long-term ATR (volatile regime)
        vol_expansion = atr_14[i] > 1.5 * atr_50[i]
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Donchian upper with volatility expansion and volume spike
            if close[i] > dc_upper[i] and vol_expansion and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower with volatility expansion and volume spike
            elif close[i] < dc_lower[i] and vol_expansion and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Donchian midline or volatility contraction
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midline
                if close[i] < dc_mid[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midline
                if close[i] > dc_mid[i]:
                    exit_signal = True
            
            # Also exit if volatility collapses (ATR contraction)
            if atr_14[i] < 0.8 * atr_50[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_DonchianBreakout_ATRVol_Volume"
timeframe = "4h"
leverage = 1.0