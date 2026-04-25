#!/usr/bin/env python3
"""
4h Donchian Breakout with 1d ATR Regime and Volume Spike Confirmation
Hypothesis: Donchian(20) breakouts capture strong trends. Use 1d ATR to filter regime - 
only trade when ATR(1d) > its 20-period MA (high volatility regimes). Add volume confirmation
(>1.5x 20-bar vol MA) to avoid false breakouts. Designed for BTC/ETH with 15-30 trades/year
to minimize fee drag while working in both bull (breakouts up) and bear (breakdowns down) markets.
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
    
    # Get 1d data for ATR regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need 20 for ATR MA + 1
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and its 20-period MA for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if i == 14:
            atr_14[i] = np.nanmean(tr[1:15])  # First ATR is average of first 14 TR
        else:
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # 20-period MA of ATR(14)
    atr_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        atr_ma_20[i] = np.mean(atr_14[i-19:i+1])
    
    # Align to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    # High volatility regime: ATR(14) > MA(20) of ATR
    high_vol_regime = atr_14_aligned > atr_ma_20_aligned
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-19:i+1])
        donchian_l[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, volume MA, and regime
    start_idx = max(20, 20)  # 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_h[i]) or 
            np.isnan(donchian_l[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(high_vol_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        h_level = donchian_h[i]
        l_level = donchian_l[i]
        vol_ma = vol_ma_20[i]
        in_high_vol = high_vol_regime[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakouts with volume confirmation in high volatility regime
            if in_high_vol and volume_confirm:
                # Long breakout: price closes above upper Donchian
                if curr_close > h_level:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: price closes below lower Donchian
                elif curr_close < l_level:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price closes below lower Donchian or volatility drops
            if curr_close < l_level or not in_high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper Donchian or volatility drops
            if curr_close > h_level or not in_high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dATR_Regime_VolumeSpike"
timeframe = "4h"
leverage = 1.0