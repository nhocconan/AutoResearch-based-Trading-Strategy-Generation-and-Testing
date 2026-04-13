#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume spike confirmation
    # Long: price > Donchian high(20) AND ATR(14) > 1.2x 50-period ATR MA AND volume > 1.5x 20-period volume MA
    # Short: price < Donchian low(20) AND ATR(14) > 1.2x 50-period ATR MA AND volume > 1.5x 20-period volume MA
    # Exit: opposite Donchian breakout or volatility contraction (ATR < 0.8x 50-period ATR MA)
    # Using 12h primary timeframe for low trade frequency, Donchian for structure,
    # ATR filter to avoid low-volatility false breakouts, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing
    atr_1d = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_1d[13] = np.nanmean(tr[1:15])  # First ATR = average of first 14 TR
        for i in range(14, len(tr)):
            if not np.isnan(atr_1d[i-1]) and not np.isnan(tr[i]):
                atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Align daily ATR to 12h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 50-period ATR MA for volatility regime filter
    atr_ma_50 = np.full(n, np.nan)
    for i in range(50, n):
        atr_ma_50[i] = np.nanmean(atr_1d_aligned[i-50:i])
    
    # Volatility filter: ATR > 1.2x 50-period ATR MA (avoid low-vol breakouts)
    vol_filter = atr_1d_aligned > (1.2 * atr_ma_50)
    
    # Get 12h data for Donchian channels and volume
    # Calculate Donchian(20) on 12h data
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: >1.5x 20-period volume MA
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_filter[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_breakout = close[i] > donch_high[i]
        short_breakout = close[i] < donch_low[i]
        
        long_entry = long_breakout and vol_filter[i] and volume_spike[i]
        short_entry = short_breakout and vol_filter[i] and volume_spike[i]
        
        # Exit conditions: opposite breakout or volatility contraction
        vol_contraction = atr_1d_aligned[i] < (0.8 * atr_ma_50[i]) if not np.isnan(atr_ma_50[i]) else False
        long_exit = short_breakout or vol_contraction
        short_exit = long_breakout or vol_contraction
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_atr_volume_v1"
timeframe = "12h"
leverage = 1.0