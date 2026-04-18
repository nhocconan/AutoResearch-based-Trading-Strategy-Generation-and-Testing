#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
Hypothesis: Trade Camarilla pivot level breaks (R1/S1) on 12h with 1d volume confirmation and ATR volatility filter. Works in bull/bear by trading breakouts from key daily pivot levels, avoiding false breakouts in low volatility. Uses 1d volume > 1.5x average and ATR(12) > 0.5*ATR(24) to filter low-volatility environments. Targets 15-25 trades/year via strict pivot break + volume + volatility confluence.
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
    
    # Get 1d data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # 1d volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # ATR filter: ATR(12) > 0.5 * ATR(24) to avoid low volatility
    def calculate_atr(high, low, close, period):
        tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]  # First TR
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_12 = calculate_atr(high, low, close, 12)
    atr_24 = calculate_atr(high, low, close, 24)
    vol_filter = atr_12 > (0.5 * atr_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 24)  # Need volume MA and ATR periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_12[i]) or np.isnan(atr_24[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter
        vol_filter_ok = vol_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume + volatility
            if close[i] > camarilla_R1_aligned[i] and vol_confirm and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + volatility
            elif close[i] < camarilla_S1_aligned[i] and vol_confirm and vol_filter_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or volatility drops
            if close[i] < camarilla_S1_aligned[i] or not vol_filter_ok:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or volatility drops
            if close[i] > camarilla_R1_aligned[i] or not vol_filter_ok:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0