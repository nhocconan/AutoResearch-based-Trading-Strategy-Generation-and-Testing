#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
Hypothesis: Trade Camarilla pivot (R1/S1) breakouts on 12h with volume confirmation and ATR-based stoploss. Camarilla levels identify key support/resistance where price often reverses or breaks. Breakouts above R1 or below S1 with volume > 1.5x average indicate institutional interest. Works in both bull and break: in uptrend, buy R1 breakouts; in downtrend, sell S1 breakouts. Uses 1d ATR for volatility filter to avoid choppy markets. Targets 15-25 trades/year via strict breakout conditions.
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
    
    # Get 1d data for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            camarilla_R1[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 12
            camarilla_S1[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 12
    
    # Calculate ATR(14) on 1d for volatility filter
    atr_period = 14
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    atr_1d = np.full_like(close_1d, np.nan)
    if len(tr) >= atr_period + 1:
        # First ATR value
        atr_1d[atr_period] = np.nanmean(tr[1:atr_period+1])
        # Wilder smoothing
        for i in range(atr_period + 1, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Align Camarilla levels and ATR to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 30)  # ensure volume MA and ATR are available
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility filter: avoid trading in extremely low volatility (chop)
        vol_filter = atr_1d_aligned[i] > 0.5 * np.nanmedian(atr_1d_aligned[max(0, i-50):i+1])
        
        if position == 0:
            # Long: price breaks above R1 + volume + volatility filter
            if close[i] > camarilla_R1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + volatility filter
            elif close[i] < camarilla_S1_aligned[i] and vol_confirm and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below S1 or ATR drops significantly (chop)
            if close[i] < camarilla_S1_aligned[i] or atr_1d_aligned[i] < 0.3 * np.nanmedian(atr_1d_aligned[max(0, i-50):i+1]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above R1 or ATR drops significantly (chop)
            if close[i] > camarilla_R1_aligned[i] or atr_1d_aligned[i] < 0.3 * np.nanmedian(atr_1d_aligned[max(0, i-50):i+1]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0