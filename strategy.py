#!/usr/bin/env python3
"""
Hypothesis: 4h strategy combining 1-day Donchian channel breakout with 1-day ATR-based volatility filter and volume confirmation.
Trades breakouts of daily high/low with confirmation from elevated volatility (ATR ratio) and volume spike.
Trades in direction of breakout only when volatility is elevated and volume confirms.
Uses tight position sizing (0.25) to limit drawdown and targets 20-40 trades/year to minimize fee drag.
Works in both bull and bear markets by capturing volatility expansion moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # 1-day ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(low_1d, 1))
    tr3 = np.abs(low_1d - np.roll(high_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_10_aligned = align_htf_to_ltf(prices, df_1d, atr_10)
    
    # ATR ratio: current ATR / 20-period ATR average (volatility expansion filter)
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_20_aligned = align_htf_to_ltf(prices, df_1d, atr_20)
    atr_ratio = atr_10_aligned / atr_20_aligned
    
    # 4h volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_ratio[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        upper_break = donchian_high_aligned[i]
        lower_break = donchian_low_aligned[i]
        vol_ratio_val = vol_ratio[i]
        atr_ratio_val = atr_ratio[i]
        
        if position == 0:
            # Enter long: price breaks above daily Donchian high + volatility expansion + volume spike
            if (price_close > upper_break and 
                atr_ratio_val > 1.2 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below daily Donchian low + volatility expansion + volume spike
            elif (price_close < lower_break and 
                  atr_ratio_val > 1.2 and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price retracement to midpoint of Donchian channel or loss of volatility/volume
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if position == 1 and price_close < midpoint:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_ATR_Volume"
timeframe = "4h"
leverage = 1.0