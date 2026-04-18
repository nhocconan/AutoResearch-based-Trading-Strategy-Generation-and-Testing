#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
Hypothesis: Breakout at Camarilla R1/S1 levels on 12h with volume confirmation and ATR filter. Works in bull by capturing breakouts above R1, in bear by capturing breakdowns below S1. Volume > 1.5x 24-period average confirms breakout strength. ATR(24) > 0.5 * ATR(96) ensures sufficient volatility. Targets 15-30 trades/year via tight Camarilla levels and volume/volatility filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Calculate Camarilla levels for previous day
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        camarilla_R1[i] = prev_close + 1.1 * range_ / 12
        camarilla_S1[i] = prev_close - 1.1 * range_ / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    # ATR filter: ATR(24) > 0.5 * ATR(96) for volatility regime
    def calculate_atr(high, low, close, period):
        atr = np.full_like(close, np.nan)
        if len(close) < period:
            return atr
        tr = np.zeros(len(close))
        tr[0] = high[0] - low[0]
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Wilder smoothing
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(close)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr
    
    atr_24 = calculate_atr(high, low, close, 24)
    atr_96 = calculate_atr(high, low, close, 96)
    vol_filter = (atr_24 > 0.5 * atr_96) & ~np.isnan(atr_24) & ~np.isnan(atr_96)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period)  # Ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_24[i]) or np.isnan(atr_96[i])):
            signals[i] = 0.0
            continue
        
        # Volume and volatility confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        vol_regime = vol_filter[i]
        
        if position == 0:
            # Long: Close > Camarilla R1 + volume + volatility
            if close[i] > camarilla_R1_aligned[i] and vol_confirm and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short: Close < Camarilla S1 + volume + volatility
            elif close[i] < camarilla_S1_aligned[i] and vol_confirm and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close < Camarilla S1 (reversal) or volatility collapse
            if close[i] < camarilla_S1_aligned[i] or not vol_regime:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close > Camarilla R1 (reversal) or volatility collapse
            if close[i] > camarilla_R1_aligned[i] or not vol_regime:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0