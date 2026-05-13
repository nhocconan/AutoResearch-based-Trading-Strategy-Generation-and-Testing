#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Power combined with 1d ADX regime filter. 
# Elder Ray measures bull/bear power via EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# In bull regime (1d ADX > 25), enter long when Bull Power turns positive after being negative.
# In bear regime (1d ADX > 25), enter short when Bear Power turns negative after being positive.
# In ranging regime (1d ADX < 20), fade extremes: long when Bear Power crosses above -ATR(10), short when Bull Power crosses below ATR(10).
# Uses discrete sizing (0.25) and 6h timeframe for low trade frequency (~12-37/year).
# Works in both bull and bear markets by adapting to regime: trend following in strong trends, mean reversion in ranges.

name = "6h_ElderRay_Power_1dADX_Regime"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        plus_dm_smooth[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
            
            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i]
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # ADX is smoothed DX
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 6h data for Elder Ray Power (EMA13 based)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate EMA13 on 6h close
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_6h - ema13_6h
    bear_power = low_6h - ema13_6h
    
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # Calculate ATR(10) on 6h for ranging regime thresholds
    def calculate_atr(high, low, close, period=10):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr10_6h = calculate_atr(high_6h, low_6h, close_6h, 10)
    atr10_6h_aligned = align_htf_to_ltf(prices, df_6h, atr10_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or \
           np.isnan(bear_power_aligned[i]) or np.isnan(atr10_6h_aligned[i]):
            signals[i] = 0.0
            continue
        
        adx = adx_1d_aligned[i]
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        atr = atr10_6h_aligned[i]
        
        # Regime determination
        is_trending = adx > 25
        is_ranging = adx < 20
        
        if position == 0:
            if is_trending:
                # Trend following: long when bull power turns positive, short when bear power turns negative
                if bull > 0 and (i == 100 or bull_power_aligned[i-1] <= 0):
                    signals[i] = 0.25
                    position = 1
                elif bear < 0 and (i == 100 or bear_power_aligned[i-1] >= 0):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Mean reversion: long when bear power crosses above -ATR, short when bull power crosses below ATR
                if bear > -atr and (i == 100 or bear_power_aligned[i-1] <= -atr):
                    signals[i] = 0.25
                    position = 1
                elif bull < atr and (i == 100 or bull_power_aligned[i-1] >= atr):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Transition regime: no action
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bear power crosses above -ATR (mean reversion) or bull power turns negative (trend exhaustion)
            if bear > -atr or bull < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bull power crosses below ATR (mean reversion) or bear power turns positive (trend exhaustion)
            if bull < atr or bear > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals