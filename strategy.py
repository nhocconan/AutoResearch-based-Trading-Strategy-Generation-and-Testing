#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 Breakout + 1d EMA34 Trend + Volume Spike.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when close breaks above H3 level AND price > 1d EMA34 AND volume > 2.0 * 12h volume MA(20);
         Short when close breaks below L3 level AND price < 1d EMA34 AND volume > 2.0 * 12h volume MA(20).
- Exit: Long exits when close crosses below L3 level; Short exits when close crosses above H3 level.
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla pivot levels from prior 1d for precise S/R, volume confirmation for participation,
  and EMA34 trend filter to avoid counter-trend trades. Proven structure with tight entries.
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
    
    # Get 1d data for Camarilla pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need sufficient data for pivots and EMA
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Camarilla: H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_H3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_L3 = close_1d - 1.125 * (high_1d - low_1d)
    
    # Calculate EMA34 for 1d trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 12h data for volume MA(20)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h volume MA to 12h timeframe (itself)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above H3 AND price > 1d EMA34 (uptrend)
                if curr_close > camarilla_H3_aligned[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close breaks below L3 AND price < 1d EMA34 (downtrend)
                elif curr_close < camarilla_L3_aligned[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when close crosses below L3
            if curr_close < camarilla_L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when close crosses above H3
            if curr_close > camarilla_H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0