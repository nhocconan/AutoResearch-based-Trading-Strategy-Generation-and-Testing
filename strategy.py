#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d ATR regime filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for ATR regime (trending vs ranging) and Camarilla pivot levels.
- Camarilla levels: H3 = close + 1.1*(high-low)/12, L3 = close - 1.1*(high-low)/12.
- ATR regime: ATR(14) > 1.5 * ATR(50) indicates trending market (breakout favorable).
- Volume confirmation: current volume > 2.0 * 20-period average volume.
- Entry: Long when price > H3 AND trending regime AND volume spike.
         Short when price < L3 AND trending regime AND volume spike.
- Exit: Opposite Camarilla level touch (price < H3 for long exit, price > L3 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets (strong breakouts) and bear markets (strong breakdowns) with regime filter avoiding false breakouts in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for ATR50
        return np.zeros(n)
    
    # True Range calculation
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr1[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # First bar
    tr2[0] = np.abs(df_1d['high'].values[0] - df_1d['close'].values[0])
    tr3[0] = np.abs(df_1d['low'].values[0] - df_1d['close'].values[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR arrays to 4h timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # H3 = close + 1.1*(high-low)/12, L3 = close - 1.1*(high-low)/12
    camarilla_h3 = df_1d['close'].values + (1.1 * (df_1d['high'].values - df_1d['low'].values) / 12)
    camarilla_l3 = df_1d['close'].values - (1.1 * (df_1d['high'].values - df_1d['low'].values) / 12)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for ATR50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(atr14_aligned[i]) or np.isnan(atr50_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Regime filter: ATR(14) > 1.5 * ATR(50) indicates trending market
        trending_regime = atr14_aligned[i] > 1.5 * atr50_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions: opposite Camarilla level touch
        if position != 0:
            # Exit long: price < H3 (breakout failed)
            if position == 1:
                if curr_close < camarilla_h3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > L3 (breakdown failed)
            elif position == -1:
                if curr_close > camarilla_l3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Camarilla breakout with regime and volume filters
        if position == 0:
            # Long: price > H3 AND trending regime AND volume spike
            long_condition = (curr_close > camarilla_h3_aligned[i] and 
                            trending_regime and
                            volume_confirm)
            
            # Short: price < L3 AND trending regime AND volume spike
            short_condition = (curr_close < camarilla_l3_aligned[i] and 
                             trending_regime and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0