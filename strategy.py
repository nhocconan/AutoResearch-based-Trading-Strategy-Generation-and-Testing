#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_ATR_Stop_v1
Hypothesis: Donchian channel breakouts with volume confirmation and ATR-based stops capture medium-term trends in BTC/ETH.
Works in bull markets via breakout follow-through and in bear markets via short breakdowns.
Volume filter reduces false breakouts. ATR stop manages risk during reversals.
Target: 25-40 trades/year by requiring high/low breaks + volume surge.
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
    
    # Donchian channels (20-period) - calculated on close prices for breakouts
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback, len(high)):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Daily ATR for volatility filter and stop calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    def calculate_atr(high, low, close, period=14):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        atr = np.full_like(tr, np.nan)
        if len(tr) >= period:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Daily volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(vol_1d, np.nan)
    vol_period = 20
    if len(vol_1d) >= vol_period:
        for i in range(vol_period, len(vol_1d)):
            vol_ma_1d[i] = np.mean(vol_1d[i-vol_period:i])
    
    # Align all daily data to 4h timeframe
    highest_high_4h = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_4h = align_htf_to_ltf(prices, df_1d, lowest_low)
    atr_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_4h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4h volume confirmation: volume > 2x 20-period average
    vol_ma_4h_local = np.full_like(volume, np.nan)
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma_4h_local[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(vol_ma_4h_local[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma_4h_local[i]
        
        if position == 0:
            # Long: price breaks above 20-period high with volume confirmation
            if close[i] > highest_high_4h[i] and vol_confirm:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 20-period low with volume confirmation
            elif close[i] < lowest_low_4h[i] and vol_confirm:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 20-period low OR ATR-based stop hit
            if close[i] < lowest_low_4h[i] or close[i] < (highest_high_4h[i] - 2.0 * atr_4h[i]):
                signals[i] = -0.30  # reverse to short
                position = -1
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price closes above 20-period high OR ATR-based stop hit
            if close[i] > highest_high_4h[i] or close[i] > (lowest_low_4h[i] + 2.0 * atr_4h[i]):
                signals[i] = 0.30  # reverse to long
                position = 1
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian_Breakout_Volume_ATR_Stop_v1"
timeframe = "4h"
leverage = 1.0