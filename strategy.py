#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Regime
Hypothesis: Donchian channel breakouts with volume confirmation and ADX regime filter capture trends while avoiding whipsaw in choppy markets. Works in bull markets by catching breakouts and in bear markets by filtering false signals via ADX < 25 (range) condition. Uses 1d ATR for volatility filtering to adapt to changing market conditions.
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
    
    # Donchian channel (20-period)
    def donchian_channel(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_high, donch_low = donchian_channel(high, low, 20)
    
    # ADX (14-period) for regime filtering
    def adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(plus_dm)
        minus_di = np.zeros_like(minus_dm)
        
        atr[period-1] = np.mean(tr[1:period+1])
        plus_di[period-1] = np.mean(plus_dm[1:period+1]) / atr[period-1] * 100 if atr[period-1] != 0 else 0
        minus_di[period-1] = np.mean(minus_dm[1:period+1]) / atr[period-1] * 100 if atr[period-1] != 0 else 0
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
            minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / atr[i] * 100 if atr[i] != 0 else 0
        
        dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
        adx_vals = np.zeros_like(dx)
        adx_vals[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx_vals[i] = (adx_vals[i-1] * (period-1) + dx[i]) / period
        
        return adx_vals
    
    adx_vals = adx(high, low, close, 14)
    
    # Volume confirmation (20-period average)
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d ATR for volatility filtering
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR (14-period)
    def atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_vals = np.zeros_like(tr)
        atr_vals[period-1] = np.mean(tr[1:period+1])
        for i in range(period, len(high)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr_1d = atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 20, 14*2)  # Donchian, volume MA20, ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(volume_ma20[i]) or 
            np.isnan(adx_vals[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # ADX regime filter: only trade when ADX > 25 (trending market)
        regime_filter = adx_vals[i] > 25
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr_1d_aligned[i] > np.percentile(atr_1d_aligned[:i+1], 20) if i >= 20 else True
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume + regime + vol filter
            if close[i] > donch_high[i] and volume_filter and regime_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + volume + regime + vol filter
            elif close[i] < donch_low[i] and volume_filter and regime_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian lower
            if close[i] < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian upper
            if close[i] > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0