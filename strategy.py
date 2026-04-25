#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeConfirm_v2
Hypothesis: Trade Camarilla R1/S1 breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation.
In bull markets: long when price breaks above R1 + 12h uptrend + volume > 1.5x avg.
In bear markets: short when price breaks below S1 + 12h downtrend + volume > 1.5x avg.
Exit on opposite Camarilla level touch or trend reversal.
Position size: 0.25 to balance risk and return.
Target: 25-40 trades/year (100-160 total over 4 years) to stay under 400-trade 4h hard max.
Uses Camarilla pivot structure for institutional levels, 12h trend for multi-timeframe alignment,
and volume filter to avoid low-conviction breakouts. Works in bull (breakouts with uptrend)
and bear (breakdowns with downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need enough data for EMA
        return np.zeros(n)
    
    # Calculate 12h EMA50 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot levels (daily calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_R1 = np.full(len(df_1d), np.nan)
    camarilla_S1 = np.full(len(df_1d), np.nan)
    camarilla_H3 = np.full(len(df_1d), np.nan)
    camarilla_L3 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0 or np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i-1]):
            continue
        # Camarilla formulas using previous day's OHLC
        diff = high_1d[i-1] - low_1d[i-1]
        camarilla_R1[i] = close_1d[i-1] + diff * 1.1 / 12
        camarilla_S1[i] = close_1d[i-1] - diff * 1.1 / 12
        camarilla_H3[i] = close_1d[i-1] + diff * 1.1 / 4
        camarilla_L3[i] = close_1d[i-1] - diff * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above EMA50)
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above R1 + 12h uptrend + volume confirmation
            long_setup = (close[i] > camarilla_R1_aligned[i]) and htf_12h_bullish and volume_confirm[i]
            
            # Short setup: price breaks below S1 + 12h downtrend + volume confirmation
            short_setup = (close[i] < camarilla_S1_aligned[i]) and htf_12h_bearish and volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches S1 (stop) OR 12h trend turns bearish
            if (close[i] <= camarilla_S1_aligned[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches R1 (stop) OR 12h trend turns bullish
            if (close[i] >= camarilla_R1_aligned[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0