#!/usr/bin/env python3
"""
4h_ADX_Donchian_Breakout_Volume
Hypothesis: Trade breakouts from Donchian(20) channels with ADX trend filter and volume confirmation.
Long when price breaks above upper band + ADX > 25 + volume > 1.5x average.
Short when price breaks below lower band + ADX > 25 + volume > 1.5x average.
ADX ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
Volume confirmation adds conviction to breakouts.
Target: 15-30 trades/year per symbol with position size 0.25.
Works in bull/bear: ADX filter avoids false breakouts in ranging markets.
"""

name = "4h_ADX_Donchian_Breakout_Volume"
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
    
    # Donchian Channel (20-period)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # ADX (14-period) - measures trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original array
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values
        atr = np.full(n, np.nan)
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        
        if len(tr) >= period:
            # Initial ATR
            atr[period-1] = np.nanmean(tr[1:period+1])
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            
            # Initial DI
            plus_di[period-1] = 100 * np.nanmean(plus_dm[1:period+1]) / atr[period-1]
            minus_di[period-1] = 100 * np.nanmean(minus_dm[1:period+1]) / atr[period-1]
            
            for i in range(period, n):
                plus_di[i] = 100 * ((plus_di[i-1] * (period-1) + plus_dm[i]) / period) / atr[i]
                minus_di[i] = 100 * ((minus_di[i-1] * (period-1) + minus_dm[i]) / period) / atr[i]
        
        # DX and ADX
        dx = np.full(n, np.nan)
        adx = np.full(n, np.nan)
        
        if len(plus_di) >= period:
            for i in range(period, n):
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            
            if len(dx) >= 2*period-1:
                adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
                for i in range(2*period-1, n):
                    adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume average (20-period)
    vol_avg = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 30)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_avg[i]) or
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x average
        vol_condition = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long: break above upper band + ADX > 25 + volume confirmation
            if close[i] > upper[i] and adx[i] > 25 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band + ADX > 25 + volume confirmation
            elif close[i] < lower[i] and adx[i] > 25 and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle of channel OR ADX weakens
            mid = (upper[i] + lower[i]) / 2
            if close[i] < mid or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle of channel OR ADX weakens
            mid = (upper[i] + lower[i]) / 2
            if close[i] > mid or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals