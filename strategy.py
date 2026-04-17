#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d volume spike + 1d ADX trend filter.
Long when price > Alligator Jaw (13-period SMA shifted 8 bars) with volume > 2.0x 20-period 1d avg volume AND 1d ADX > 25.
Short when price < Alligator Jaw with same conditions.
Exit on opposite Jaw touch or ADX < 20 (trend weakening).
Designed for low-frequency, high-conviction trades in both bull and bear markets.
Target: 12-25 trades/year per symbol (50-100 total over 4 years).
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
    
    # Get 1d data for volume MA, Alligator, and ADX
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 20-period volume moving average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Williams Alligator: Jaw (13-period SMA shifted 8), Teeth (8-period SMA shifted 5), Lips (5-period SMA shifted 3)
    def calculate_alligator(high, low, close):
        typical_price = (high + low + close) / 3
        jaw = pd.Series(typical_price).rolling(window=13, min_periods=13).mean().shift(8).values
        teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().shift(5).values
        lips = pd.Series(typical_price).rolling(window=5, min_periods=5).mean().shift(3).values
        return jaw, teeth, lips
    
    jaw_1d, teeth_1d, lips_1d = calculate_alligator(high_1d, low_1d, close_1d)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(jaw_1d_aligned[i]) or 
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period 1d average volume
        volume_confirmed = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx_14_aligned[i] > 25
        trend_weakening = adx_14_aligned[i] < 20  # exit condition
        
        if position == 0:
            # Long: price > Jaw with volume confirmation and trending market
            if (close[i] > jaw_1d_aligned[i] and volume_confirmed and trending):
                signals[i] = 0.25
                position = 1
            # Short: price < Jaw with volume confirmation and trending market
            elif (close[i] < jaw_1d_aligned[i] and volume_confirmed and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches Jaw OR trend weakening
            if (close[i] <= jaw_1d_aligned[i]) or trend_weakening:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches Jaw OR trend weakening
            if (close[i] >= jaw_1d_aligned[i]) or trend_weakening:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Volume_1dADX25_Trend"
timeframe = "12h"
leverage = 1.0