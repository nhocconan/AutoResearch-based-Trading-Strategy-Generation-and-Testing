# 4h Donchian Breakout with Volume Confirmation and ADX Trend Filter
# Hypothesis: Donchian channel breakouts capture trend continuations. Volume confirms breakout strength.
# ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
# Works in both bull and bear markets by trading breakouts in the direction of the trend.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ADX for trend strength (14-period)
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values (14-period)
    def smooth_rma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = smooth_rma(tr, 14)
    plus_di = 100 * smooth_rma(plus_dm, 14) / atr
    minus_di = 100 * smooth_rma(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smooth_rma(dx, 14)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx[i]) or np.isnan(vol_ma_20[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.3 * vol_ma
        
        # ADX trend filter: only trade when ADX > 25 (trending market)
        trend_ok = adx[i] > 25
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume and trend
            if price > high_20[i] and volume_ok and trend_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume and trend
            elif price < low_20[i] and volume_ok and trend_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to middle of Donchian channel or opposite break
            mid = (high_20[i] + low_20[i]) / 2
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to middle of Donchian channel or opposite break
            mid = (high_20[i] + low_20[i]) / 2
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals