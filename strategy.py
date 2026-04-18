#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with volume confirmation on 12h timeframe.
# Uses daily Donchian channels for trend direction and volatility regime.
# Enters on 12h breakouts above/below daily Donchian levels with volume confirmation.
# Designed for fewer trades (target 20-50/year) to avoid fee drag in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian Channel (20-period) on daily data
    donchian_up = np.full(len(high_1d), np.nan)
    donchian_dn = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donchian_up[i] = np.max(high_1d[i-20:i])
        donchian_dn[i] = np.min(low_1d[i-20:i])
    
    # Align daily Donchian to 12h timeframe
    donchian_up_aligned = align_htf_to_ltf(prices, df_1d, donchian_up)
    donchian_dn_aligned = align_htf_to_ltf(prices, df_1d, donchian_dn)
    
    # Calculate ATR(14) on 12h for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # need daily Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_up_aligned[i]) or np.isnan(donchian_dn_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above daily Donchian upper with volume
            if close[i] > donchian_up_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below daily Donchian lower with volume
            elif close[i] < donchian_dn_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below daily Donchian lower or ATR-based stop
            if close[i] < donchian_dn_aligned[i] or close[i] < open_price[i] - 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily Donchian upper or ATR-based stop
            if close[i] > donchian_up_aligned[i] or close[i] > open_price[i] + 2.0 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20Daily_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0