#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility regime filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ATR(14) to measure volatility regime - only trade when ATR(14) > ATR(50) (expanding volatility).
- Donchian channel: 20-period high/low from prior 4h candle (using prior close to avoid look-ahead).
- Entry: Long when price breaks above prior Donchian high AND ATR regime expanding AND volume > 1.5 * volume MA(20).
         Short when price breaks below prior Donchian low AND ATR regime expanding AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below prior Donchian low,
        exit short when price crosses above prior Donchian high.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures breakouts during expanding volatility regimes, designed to work in both bull and bear markets by filtering for genuine momentum bursts.
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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR(50) for volatility regime filter
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = df_1d_high - df_1d_low
    tr2 = np.abs(df_1d_high - np.roll(df_1d_close, 1))
    tr3 = np.abs(df_1d_low - np.roll(df_1d_close, 1))
    tr1[0] = df_1d_high[0] - df_1d_low[0]  # First bar
    tr2[0] = np.abs(df_1d_high[0] - df_1d_close[0])
    tr3[0] = np.abs(df_1d_low[0] - df_1d_close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculations
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Regime: expanding volatility when ATR(14) > ATR(50)
    atr_regime = atr_14 > atr_50
    
    # Calculate prior 4h Donchian(20) levels (using prior candle to avoid look-ahead)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 60)  # Need enough bars for ATR(50) and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_regime_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold) and ATR regime
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            regime_expanding = atr_regime_aligned[i] > 0.5
            
            # Long: Price breaks above prior Donchian high AND ATR regime expanding AND volume confirmed
            if curr_close > donchian_high[i] and regime_expanding and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior Donchian low AND ATR regime expanding AND volume confirmed
            elif curr_close < donchian_low[i] and regime_expanding and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior Donchian low (breakdown)
            if curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior Donchian high (breakout)
            if curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_Regime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0