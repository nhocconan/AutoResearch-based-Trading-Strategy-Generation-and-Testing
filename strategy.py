#!/usr/bin/env python3
# 6h_adx_alligator_12h_trend_v1
# Hypothesis: 6h strategy using Williams Alligator (Jaw/Teeth/Lips) for trend state,
# filtered by 12h ADX > 25 for strong trend confirmation. Entries on Alligator alignment
# with 12h trend direction. Exits on trend weakness or Alligator crossover reversal.
# Works in bull/bear: 12h ADX filters ranging markets, Alligator captures momentum,
# ADX > 25 ensures we only trade strong trends reducing whipsaw.
# Target: 12-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_alligator_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h HTF data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h ADX calculation
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Williams Alligator on 6h
    # Jaw (blue): 13-period SMMA smoothed 8 bars ahead
    # Teeth (red): 8-period SMMA smoothed 5 bars ahead  
    # Lips (green): 5-period SMMA smoothed 3 bars ahead
    def smma(data, period):
        sma = np.zeros_like(data)
        sma[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smma(smma(close, 13), 8)  # SMMA(13) then smoothed 8
    teeth = smma(smma(close, 8), 5)  # SMMA(8) then smoothed 5
    lips = smma(smma(close, 5), 3)   # SMMA(5) then smoothed 3
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if np.isnan(adx_12h_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ADX weakens (<20) OR Alligator reverses (lips < teeth)
            if adx_12h_aligned[i] < 20 or lips[i] < teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX weakens (<20) OR Alligator reverses (lips > teeth)
            if adx_12h_aligned[i] < 20 or lips[i] > teeth[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: strong trend (ADX > 25) + Alligator alignment
            if adx_12h_aligned[i] > 25:
                # Bullish alignment: Lips > Teeth > Jaw
                if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish alignment: Lips < Teeth < Jaw
                elif lips[i] < teeth[i] and teeth[i] < jaw[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals