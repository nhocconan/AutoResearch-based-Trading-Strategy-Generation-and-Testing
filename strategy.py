#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 Breakout + 1d ADX Trend + Volume Spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ADX(14) for trend filter (ADX > 25 = trending, ADX < 20 = ranging).
- Entry: Long when close breaks above H3 level AND ADX > 25 AND volume > 2.0 * 4h volume MA(20);
         Short when close breaks below L3 level AND ADX > 25 AND volume > 2.0 * 4h volume MA(20).
- Exit: Long exits when close crosses below L3 level; Short exits when close crosses above H3 level.
- Signal size: 0.25 discrete to control fee drag.
- Uses Camarilla pivot levels from prior 1d for precise S/R, ADX trend filter to ensure trending markets,
  and volume confirmation for participation. Proven structure with tight entries.
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
    
    # Get 1d data for Camarilla pivot and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for ADX(14)
        return np.zeros(n)
    
    # Calculate ADX for 1d trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low),
    # L3 = close - 1.125*(high-low), L4 = close - 1.5*(high-low)
    camarilla_H3 = close_1d + 1.125 * (high_1d - low_1d)
    camarilla_L3 = close_1d - 1.125 * (high_1d - low_1d)
    
    # Align 1d indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # Get 4h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # ADX needs 30, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm and adx_aligned[i] > 25:
                # Long: Close breaks above H3 AND ADX > 25 (strong trend)
                if curr_close > camarilla_H3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Close breaks below L3 AND ADX > 25 (strong trend)
                elif curr_close < camarilla_L3_aligned[i]:
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

name = "4h_Camarilla_H3L3_Breakout_1dADX_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0