#!/usr/bin/env python3
"""
Hypothesis: 6h Volume-Weighted RSI + 1d ADX Regime Filter.
Long when VWRSI(14) < 30 (oversold) AND 1d ADX > 25 (trending market).
Short when VWRSI(14) > 70 (overbought) AND 1d ADX > 25 (trending market).
Exit when VWRSI returns to neutral (40-60 range) OR 1d ADX < 20 (range market).
Uses 1d ADX for regime detection to avoid whipsaws in ranging markets, 
while 6h VWRSI provides precise entry timing during trends.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX for regime filter (trending vs ranging)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros(len(high))
        atr[period] = np.nansum(tr[1:period+1]) if period < len(tr) else 0
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        
        for i in range(period, len(high)):
            if atr[i] > 0:
                plus_di[i] = 100 * (plus_dm[i] / atr[i])
                minus_di[i] = 100 * (minus_dm[i] / atr[i])
                if (plus_di[i] + minus_di[i]) > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(len(high))
        adx[2*period-1] = np.nanmean(dx[period:2*period]) if 2*period <= len(dx) else 0
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Volume-Weighted RSI on 6h timeframe
    def calculate_vw_rsi(close, volume, period=14):
        delta = np.diff(close, prepend=close[0])
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        
        # Volume-weighted gains and losses
        vol_up = up * volume
        vol_down = down * volume
        
        # Wilder's smoothing with volume weighting
        avg_vol_up = np.zeros(len(close))
        avg_vol_down = np.zeros(len(close))
        
        avg_vol_up[period] = np.nansum(vol_up[1:period+1]) if period < len(vol_up) else 0
        avg_vol_down[period] = np.nansum(vol_down[1:period+1]) if period < len(vol_down) else 0
        
        for i in range(period+1, len(close)):
            avg_vol_up[i] = (avg_vol_up[i-1] * (period-1) + vol_up[i]) / period
            avg_vol_down[i] = (avg_vol_down[i-1] * (period-1) + vol_down[i]) / period
        
        rs = np.zeros(len(close))
        rsi = np.zeros(len(close))
        
        for i in range(period, len(close)):
            if avg_vol_down[i] > 0:
                rs[i] = avg_vol_up[i] / avg_vol_down[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100 if avg_vol_up[i] > 0 else 50
        
        return rsi
    
    vwrsi_6h = calculate_vw_rsi(close, volume, 14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(vwrsi_6h[i]):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        vwrsi = vwrsi_6h[i]
        
        if position == 0:
            # Long: VWRSI oversold (<30) AND trending market (ADX > 25)
            if vwrsi < 30 and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short: VWRSI overbought (>70) AND trending market (ADX > 25)
            elif vwrsi > 70 and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VWRSI returns to neutral (40-60) OR market becomes ranging (ADX < 20)
            if vwrsi >= 40 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VWRSI returns to neutral (40-60) OR market becomes ranging (ADX < 20)
            if vwrsi <= 60 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolumeWeightedRSI_1dADX_Regime"
timeframe = "6h"
leverage = 1.0