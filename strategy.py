#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted RSI with 1d ADX regime filter
# VW-RSI(14) uses typical price * volume to weight RSI calculation
# Long when VW-RSI < 30 in ranging market (ADX < 20) or VW-RSI > 50 in uptrend (ADX > 25)
# Short when VW-RSI > 70 in ranging market or VW-RSI < 50 in downtrend
# Volume weighting reduces false signals during low-volume periods
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_VolWeightedRSI_1dADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14) from prior completed 1d bar
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = (plus_dm[i] / atr[i]) * 100
                minus_di[i] = (minus_dm[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_shifted = np.roll(adx_14, 1)
    adx_14_shifted[0] = np.nan
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14_shifted)
    
    # Calculate Volume-Weighted RSI (14) on 6h data
    typical_price = (high + low + close) / 3.0
    vp = typical_price * volume  # volume-weighted price
    
    # Price changes
    delta = vp - np.roll(vp, 1)
    delta[0] = 0
    
    # Separate gains and losses
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.zeros_like(avg_gain)
    rs[avg_loss != 0] = avg_gain[avg_loss != 0] / avg_loss[avg_loss != 0]
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_14_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entries
            if adx_14_aligned[i] < 20:  # Ranging market
                if rsi[i] < 30:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70:  # Overbought
                    signals[i] = -0.25
                    position = -1
            else:  # Trending market (ADX >= 20)
                if adx_14_aligned[i] > 25:  # Strong trend
                    if rsi[i] > 50 and close[i] > np.roll(close, 1)[i]:  # Uptrend continuation
                        signals[i] = 0.25
                        position = 1
                    elif rsi[i] < 50 and close[i] < np.roll(close, 1)[i]:  # Downtrend continuation
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Exit long: RSI > 70 in ranging OR RSI < 40 in trending
            if (adx_14_aligned[i] < 20 and rsi[i] > 70) or (adx_14_aligned[i] >= 20 and rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 30 in ranging OR RSI > 60 in trending
            if (adx_14_aligned[i] < 20 and rsi[i] < 30) or (adx_14_aligned[i] >= 20 and rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals