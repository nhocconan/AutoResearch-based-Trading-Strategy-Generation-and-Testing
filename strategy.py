#!/usr/bin/env python3
"""
4h_Donchian_Breakout_20_With_Volume_and_ADX_Trend
Hypothesis: Donchian channel breakouts capture strong momentum moves. 
Filtering by ADX > 25 ensures we only trade in trending markets, 
while volume confirmation (>1.5x 20-bar average) avoids false breakouts. 
The 20-period Donchian channel provides clear entry/exit levels. 
Designed for 4h timeframe to target 20-50 trades per year, minimizing fee drag.
Works in both bull and bear regimes by only taking breakouts in the 
direction of the ADX-confirmed trend.
"""

name = "4h_Donchian_Breakout_20_With_Volume_and_ADX_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX calculation (14-period) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            elif minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
                
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum / period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum / period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
        
        # Calculate DX and ADX
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        for i in range(2*period, len(high)):
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # Smooth DX to get ADX
        adx[2*period] = np.mean(dx[period+1:2*period+1])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    # Calculate ADX
    adx = calculate_adx(high, low, close, 14)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after enough data for indicators
        if position == 0:
            # LONG: break above Donchian high with volume spike and ADX > 25 (uptrend)
            if (close[i] > donchian_high[i] and 
                volume_spike[i] and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # SHORT: break below Donchian low with volume spike and ADX > 25 (downtrend)
            elif (close[i] < donchian_low[i] and 
                  volume_spike[i] and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price drops below Donchian low or ADX drops below 20 (trend weakening)
            if (close[i] < donchian_low[i] or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price rises above Donchian high or ADX drops below 20 (trend weakening)
            if (close[i] > donchian_high[i] or 
                adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals