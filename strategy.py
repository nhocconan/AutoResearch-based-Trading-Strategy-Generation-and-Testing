#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout + volume confirmation + ADX(14) trend filter
# Donchian channels identify structural breakouts with clear support/resistance levels.
# Volume confirmation ensures institutional participation in the breakout.
# ADX(14) > 25 filters for trending markets to avoid false breakouts in ranging conditions.
# Designed for low trade frequency (20-50/year) to minimize fee drag while maintaining edge.
# Works in both bull and bear markets by capturing strong directional moves with confirmation.

name = "4h_Donchian20_Breakout_Volume_ADXTrend_v1"
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
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX(14) for trend filter
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian and ADX
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_spike = volume_spike[i]
        curr_adx = adx[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and ADX > 25 (trending market)
            if curr_volume_spike and curr_adx > 25:
                # Bullish entry: price breaks above Donchian upper band
                if curr_close > curr_highest_high:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Donchian lower band
                elif curr_close < curr_lowest_low:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks Donchian lower band
            # Calculate ATR for stoploss
            tr_i = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr_approx = tr_i  # Simplified ATR approximation for stop
            if curr_close < entry_price - 2.0 * atr_approx:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks Donchian upper band
            tr_i = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr_approx = tr_i
            if curr_close > entry_price + 2.0 * atr_approx:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_highest_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals