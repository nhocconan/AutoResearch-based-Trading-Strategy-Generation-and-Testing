#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and ADX Trend Filter
Long: Price breaks above Donchian(20) high + volume > 2x 20-period volume MA + ADX > 25
Short: Price breaks below Donchian(20) low + volume > 2x 20-period volume MA + ADX > 25
Exit: Opposite Donchian break or ADX < 20 (trend weakens)
ATR-based stop: exit if price moves against position by 2.5x ATR(20)
Target: 20-30 trades/year per symbol
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
    
    # Donchian channels (20-period high/low)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 20-period volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        if atr[period] > 0:
            plus_dm_sm = np.zeros_like(high)
            minus_dm_sm = np.zeros_like(high)
            plus_dm_sm[period] = np.mean(plus_dm[1:period+1])
            minus_dm_sm[period] = np.mean(minus_dm[1:period+1])
            
            for i in range(period+1, len(high)):
                plus_dm_sm[i] = (plus_dm_sm[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_sm[i] = (minus_dm_sm[i-1] * (period-1) + minus_dm[i]) / period
                
                plus_di[i] = 100 * plus_dm_sm[i] / atr[i]
                minus_di[i] = 100 * minus_dm_sm[i] / atr[i]
                dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # ADX is smoothed DX
        adx = np.zeros_like(high)
        adx[2*period] = np.mean(dx[period+1:2*period+1]) if (2*period+1) <= len(dx) else 0
        for i in range(2*period+1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # ATR for stop loss (20-period)
    def calculate_atr(high, low, close, period=20):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 20)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 40  # warmup for ADX/ATR
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Exit conditions first
        if position == 1:
            # Long exit: price breaks below Donchian low OR ADX weakens OR ATR stop
            if price < lowest_low[i] or adx[i] < 20 or price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian high OR ADX weakens OR ATR stop
            if price > highest_high[i] or adx[i] < 20 or price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        else:  # position == 0
            # Long entry: break above Donchian high + volume spike + strong trend
            if price > highest_high[i] and vol > 2.0 * vol_ma[i] and adx[i] > 25:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: break below Donchian low + volume spike + strong trend
            elif price < lowest_low[i] and vol > 2.0 * vol_ma[i] and adx[i] > 25:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_VolumeSpike_ADX25_ATRStop"
timeframe = "4h"
leverage = 1.0