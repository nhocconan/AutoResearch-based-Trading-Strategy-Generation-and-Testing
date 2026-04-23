#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation
- Long when price breaks above 20-period Donchian high AND ATR(14) > 1d ATR(14) MA AND volume > 1.5x 20-period average
- Short when price breaks below 20-period Donchian low AND ATR(14) > 1d ATR(14) MA AND volume > 1.5x 20-period average
- Exit when price crosses 10-period Donchian midpoint (mean reversion)
- Uses 1d ATR(14) for volatility regime filter to ensure breakouts occur in sufficient volatility
- Volume confirmation reduces false breakouts
- Designed for both bull and bear markets: volatility filter avoids low-volatility false signals
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for ATR filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = np.zeros(len(close_1d))
    atr_1d[14] = np.mean(tr_1d[:14])
    for i in range(15, len(close_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i-1]) / 14
    
    # Calculate 1d ATR(14) moving average (20-period)
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 20, 34)  # Need 20 for Donchian, 20 for volume MA, 34 for ATR stability
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate current 12h ATR(14) for volatility filter
        if i >= 14:
            tr1_12h = np.abs(high[i] - low[i])
            tr2_12h = np.abs(high[i] - close[i-1]) if i > 0 else 0
            tr3_12h = np.abs(low[i] - close[i-1]) if i > 0 else 0
            tr_12h = max(tr1_12h, tr2_12h, tr3_12h)
            # Simplified ATR calculation for current bar (using smoothed value)
            atr_12h = tr_12h  # Use current TR as proxy for volatility filter
        else:
            atr_12h = 0
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]  # Break above Donchian high
        breakout_down = close[i] < lowest_low[i]  # Break below Donchian low
        
        # Volatility filter: current 12h TR > 1d ATR MA (ensures sufficient volatility)
        volatility_ok = atr_12h > atr_ma_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + volatility + volume confirmation
            if breakout_up and volatility_ok and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volatility + volume confirmation
            elif breakout_down and volatility_ok and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 10-period Donchian midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midpoint
                if close[i] < donchian_mid[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above midpoint
                if close[i] > donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0