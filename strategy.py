#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for volatility regime (high volatility = trend follow, low volatility = avoid).
- Donchian channels: 20-period high/low from 12h data for breakout signals.
- Entry: Long when price breaks above prior 20-period 12h high AND 1d ATR(14) > ATR(50) AND volume > 1.5 * volume MA(20).
         Short when price breaks below prior 20-period 12h low AND 1d ATR(14) > ATR(50) AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below prior 10-period 12h low,
        exit short when price crosses above prior 10-period 12h high.
- Signal size: 0.25 discrete to balance return and drawdown.
Uses volatility regime filter to avoid whipsaws in ranging markets and capture strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Prior 20-period high/low for breakout (avoid look-ahead)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Prior 10-period high/low for exit (avoid look-ahead)
    donchian_exit_high = pd.Series(high_12h).rolling(window=10, min_periods=10).max().shift(1).values
    donchian_exit_low = pd.Series(low_12h).rolling(window=10, min_periods=10).min().shift(1).values
    
    # Calculate 1d ATR for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF indicators to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_exit_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_exit_high)
    donchian_exit_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_exit_low)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # Calculate volume MA(20) for confirmation (using 12h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 60)  # Need enough bars for ATR50 and Donchian20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_exit_high_aligned[i]) or np.isnan(donchian_exit_low_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold) and volatility regime
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            high_volatility = atr_14_aligned[i] > atr_50_aligned[i]  # ATR(14) > ATR(50) = trending regime
            
            # Long: Price breaks above prior 20-period 12h high AND high volatility AND volume confirmed
            if curr_close > donchian_high_aligned[i] and high_volatility and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 20-period 12h low AND high volatility AND volume confirmed
            elif curr_close < donchian_low_aligned[i] and high_volatility and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior 10-period 12h low
            if curr_close < donchian_exit_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior 10-period 12h high
            if curr_close > donchian_exit_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_Regime_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0