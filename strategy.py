#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for regime filter (low volatility when ATR ratio < 0.8, high volatility when > 1.2).
- Donchian channels: 20-period high/low from 6h data for breakout detection.
- Entry: Long when price breaks above 20-period Donchian high AND ATR ratio < 0.8 (low vol) AND volume > 1.5 * volume MA(20).
         Short when price breaks below 20-period Donchian low AND ATR ratio < 0.8 (low vol) AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below 20-period Donchian mid,
        exit short when price crosses above 20-period Donchian mid.
- Signal size: 0.25 discrete to balance return and drawdown.
Uses ATR regime filter to avoid whipsaws in high volatility and capture breakouts in low volatility environments.
Works in both bull and bear markets by focusing on volatility contraction/expansion cycles.
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
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate ATR ratio: current ATR / 50-period MA of ATR (regime detection)
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma_50
    
    # Align HTF indicators to 6h
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate Donchian channels (20-period) from 6h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Need enough bars for Donchian and ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold) and low volatility regime
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            low_vol_regime = atr_ratio_aligned[i] < 0.8  # Low volatility environment
            
            # Long: Price breaks above Donchian high AND low vol regime AND volume confirmed
            if curr_close > donchian_high[i] and low_vol_regime and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND low vol regime AND volume confirmed
            elif curr_close < donchian_low[i] and low_vol_regime and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below Donchian mid (mean reversion)
            if curr_close < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above Donchian mid (mean reversion)
            if curr_close > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1dATR_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0