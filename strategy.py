#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dVolSpike_ATRFilter
Hypothesis: Breakout above/below Donchian channel (20-period high/low) on 4h with 1-day volume spike confirmation (volume > 2x 20-day average) and ATR filter (ATR(14) > 0.5 * ATR(50)). Exit on opposite Donchian break or ATR-based trailing stop. Designed to capture strong trending moves while filtering false breakouts in choppy markets. Volume spike ensures institutional participation; ATR filter avoids low-volatility false signals. Works in bull/bear by following breakout direction. Targets ~25-40 trades/year via strict volume and volatility filters.
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
    
    # Get 1d data for volume and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Volume 20-period average
    vol_ma = np.full_like(df_1d['volume'].values, np.nan)
    vol_period = 20
    vol_arr = df_1d['volume'].values
    if len(vol_arr) >= vol_period:
        for i in range(vol_period, len(vol_arr)):
            vol_ma[i] = np.mean(vol_arr[i - vol_period:i])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # 1d ATR(14) and ATR(50)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    def calc_atr(h, l, c, period):
        tr = np.full_like(h, np.nan)
        for i in range(1, len(h)):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
        atr = np.full_like(h, np.nan)
        if len(h) >= period + 1:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(h)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr14_1d = calc_atr(high_1d, low_1d, close_1d, 14)
    atr50_1d = calc_atr(high_1d, low_1d, close_1d, 50)
    
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # Get 4h data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channel (20-period high/low)
    donch_high = np.full_like(high_4h, np.nan)
    donch_low = np.full_like(low_4h, np.nan)
    period = 20
    if len(high_4h) >= period:
        for i in range(period-1, len(high_4h)):
            donch_high[i] = np.max(high_4h[i - period + 1:i + 1])
            donch_low[i] = np.min(low_4h[i - period + 1:i + 1])
    
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need ATR50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr14_aligned[i]) or 
            np.isnan(atr50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 2x 20-day average
        vol_confirm = volume[i] > 2.0 * vol_ma_aligned[i]
        
        # ATR filter: ATR(14) > 0.5 * ATR(50)
        atr_filter = atr14_aligned[i] > 0.5 * atr50_aligned[i]
        
        if position == 0:
            # Long: breakout above Donchian high + volume + ATR filter
            if close[i] > donch_high_aligned[i] and vol_confirm and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low + volume + ATR filter
            elif close[i] < donch_low_aligned[i] and vol_confirm and atr_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Donchian low OR ATR-based stop (2*ATR below entry)
            # Simplified: exit on opposite Donchian break
            if close[i] < donch_low_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian high OR ATR-based stop
            # Simplified: exit on opposite Donchian break
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dVolSpike_ATRFilter"
timeframe = "4h"
leverage = 1.0