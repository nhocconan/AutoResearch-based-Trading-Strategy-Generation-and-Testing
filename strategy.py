#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + ADX Trend Filter
Uses Donchian(20) breakouts confirmed by volume spikes and ADX trend strength.
Designed for low trade frequency (target: 15-40 trades/year) with strong edge in trending markets.
Works in both bull and bear markets by only taking trades in the direction of the ADX trend.
"""

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
    
    # Donchian channels (20-period)
    lookback = 20
    dc_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    dc_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ADX for trend strength (14-period)
    adx_period = 14
    tr1 = pd.Series(high).rolling(window=2).max() - pd.Series(low).rolling(window=2).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=adx_period, min_periods=adx_period).mean()
    
    plus_dm = pd.Series(high).diff()
    minus_dm = pd.Series(low).diff().abs()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    plus_di = 100 * (plus_dm.rolling(window=adx_period, min_periods=adx_period).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(window=adx_period, min_periods=adx_period).sum() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # enough for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = dc_high[i]
        lower = dc_low[i]
        vol_spike = volume_spike[i]
        trend_strength = adx[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and strong uptrend (ADX > 25)
            if (price > upper and 
                vol_spike and 
                trend_strength > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and strong downtrend (ADX > 25)
            elif (price < lower and 
                  vol_spike and 
                  trend_strength > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price re-enters Donchian channel or trend weakens
            if price < upper or trend_strength < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price re-enters Donchian channel or trend weakens
            if price > lower or trend_strength < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0