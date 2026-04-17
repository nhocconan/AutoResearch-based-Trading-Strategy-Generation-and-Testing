#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray + 1d ADX/EMA Regime Filter.
Long when Bull Power > 0 and Bear Power < 0 in bull regime (ADX > 25 and price > EMA50).
Short when Bear Power < 0 and Bull Power > 0 in bear regime (ADX > 25 and price < EMA50).
Exit when power diverges or regime shifts to ranging (ADX < 20).
Uses 13-period Elder Ray (EMA13) for sensitivity. Target: 50-150 total trades over 4 years (12-37/year).
Works in both bull (trend continuation) and bear (mean reversion at extremes) via regime filter.
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
    
    # Get 1d data for ADX and EMA regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros_like(close)
        dm_minus = np.zeros_like(close)
        for i in range(1, len(close)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            dm_plus[i] = up_move if up_move > down_move and up_move > 0 else 0
            dm_minus[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        # Smoothed TR, DM+, DM- (Wilder's smoothing)
        atr = np.zeros_like(close)
        dmp = np.zeros_like(close)
        dmm = np.zeros_like(close)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        dmp[period] = np.mean(dm_plus[1:period+1])
        dmm[period] = np.mean(dm_minus[1:period+1])
        
        # Wilder's smoothing
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dmp[i] = (dmp[i-1] * (period-1) + dm_plus[i]) / period
            dmm[i] = (dmm[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        dip = np.zeros_like(close)
        dim = np.zeros_like(close)
        dx = np.zeros_like(close)
        for i in range(period, len(close)):
            if atr[i] > 0:
                dip[i] = dmp[i] / atr[i] * 100
                dim[i] = dmm[i] / atr[i] * 100
                if dip[i] + dim[i] > 0:
                    dx[i] = abs(dip[i] - dim[i]) / (dip[i] + dim[i]) * 100
                else:
                    dx[i] = 0
        
        # ADX (smoothed DX)
        adx = np.zeros_like(close)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    # Calculate 1d EMA50
    def calculate_ema(data, period):
        ema = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    ema50_1d = calculate_ema(close_1d, 50)
    
    # Calculate 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    def calculate_ema_fast(data, period):
        ema = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        return ema
    
    ema13 = calculate_ema_fast(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Align 1d indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_1d_aligned[i]
        ema50 = ema50_1d_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        price = close[i]
        
        # Regime definition
        is_bull_regime = adx_val > 25 and price > ema50
        is_bear_regime = adx_val > 25 and price < ema50
        is_ranging = adx_val < 20
        
        if position == 0:
            # Long: Bull Power > 0 and Bear Power < 0 in bull regime
            if bp > 0 and br < 0 and is_bull_regime:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and Bull Power > 0 in bear regime
            elif br < 0 and bp > 0 and is_bear_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Power diverges OR regime shifts to ranging/bear
            if bp <= 0 or br >= 0 or is_ranging or is_bear_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Power diverges OR regime shifts to ranging/bull
            if br >= 0 or bp <= 0 or is_ranging or is_bull_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADXEMA_Regime"
timeframe = "6h"
leverage = 1.0