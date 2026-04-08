#!/usr/bin/env python3
"""
6h ADX + Volume Breakout with 1d Trend Filter
Hypothesis: ADX > 25 identifies strong trends; breakouts in trend direction with volume surge capture momentum.
Works in bull/bear by using 1d EMA trend filter to only take long in uptrend, short in downtrend.
Volume surge confirms institutional participation. Target: 20-30 trades/year on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_vol_breakout_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
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
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        atr[period] = np.nansum(tr[1:period+1])
        plus_di[period] = 100 * np.nansum(plus_dm[1:period+1]) / atr[period] if atr[period] != 0 else 0
        minus_di[period] = 100 * np.nansum(minus_dm[1:period+1]) / atr[period] if atr[period] != 0 else 0
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = 100 * plus_dm[i] / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm[i] / atr[i] if atr[i] != 0 else 0
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # Smooth DX to get ADX
        adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    # Calculate ADX
    adx = calculate_adx(high, low, close, 14)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_surge[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ADX weakens OR trend turns bearish
            if (adx[i] < 20 or 
                close[i] <= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ADX weakens OR trend turns bullish
            if (adx[i] < 20 or 
                close[i] >= ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: ADX strong + price breaks above recent high + volume surge + uptrend
            if (adx[i] > 25 and 
                close[i] >= np.max(high[max(0, i-10):i]) and
                close[i] > ema_50_1d_aligned[i] and
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short: ADX strong + price breaks below recent low + volume surge + downtrend
            elif (adx[i] > 25 and 
                  close[i] <= np.min(low[max(0, i-10):i]) and
                  close[i] < ema_50_1d_aligned[i] and
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals