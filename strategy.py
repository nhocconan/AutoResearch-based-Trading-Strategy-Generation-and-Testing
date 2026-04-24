#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for ATR(14) filter and Donchian channel calculation (from prior 1d bar).
- Entry: Long when price breaks above Donchian(20) upper AND ATR(14) > 1.5 * ATR(14) MA(50) AND volume > 1.5 * 12h volume MA(20);
         Short when price breaks below Donchian(20) lower AND ATR(14) > 1.5 * ATR(14) MA(50) AND volume > 1.5 * 12h volume MA(20).
- Exit: Close-based reversal (opposite signal) or stoploss via ATR trailing (implemented as signal=0 when price closes below/above Donchian midpoint).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ATR filter ensures breakouts occur during expanded volatility, reducing false signals in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior 1d Donchian(20) channels (use shift(1) to avoid look-ahead)
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20)
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20)
    donchian_upper = high_roll.max().shift(1)
    donchian_lower = low_roll.min().shift(1)
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Calculate ATR(14) on 1d
    tr1 = pd.Series(high_1d) - pd.Series(low_1d)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_ma_50 = atr_14.rolling(window=50, min_periods=50).mean()
    
    # Get 12h data for volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper.values)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower.values)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid.values)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14.values)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50.values)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (max of 50 for ATR MA, 20 for Donchian/vol MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_ma_50_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Stoploss: exit if price closes below/above Donchian midpoint
        if position == 1:
            if curr_close < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            if curr_close > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Breakout conditions with volume confirmation and ATR filter
        bullish_breakout = curr_close > donchian_upper_aligned[i]
        bearish_breakout = curr_close < donchian_lower_aligned[i]
        
        # ATR filter: current ATR > 1.5 * ATR MA(50)
        atr_expansion = atr_14_aligned[i] > 1.5 * atr_ma_50_aligned[i]
        
        # Volume confirmation
        vol_confirm = curr_volume > 1.5 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if atr_expansion and vol_confirm:
                # Long: bullish breakout
                if bullish_breakout:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish breakout
                elif bearish_breakout:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_ATRFilter_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0