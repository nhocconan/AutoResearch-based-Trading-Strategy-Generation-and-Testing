#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility regime filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for volatility regime (high volatility when ATR > 1.5 * ATR MA(50)).
- Donchian channels: 20-period high/low breakouts for entry.
- Entry: Long when price breaks above Donchian(20) high AND high volatility regime AND volume > 2.0 * volume MA(20).
         Short when price breaks below Donchian(20) low AND high volatility regime AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below Donchian(20) low,
        exit short when price crosses above Donchian(20) high.
- Signal size: 0.25 discrete to control drawdown and minimize fee churn.
Designed to capture explosive moves in both bull and bear markets via volatility expansion and breakout structure.
Proven pattern from DB: Donchian breakouts with volatility and volume filters show strong test performance.
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
    
    # Get 1d data for ATR volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR MA(50) for regime filter
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # High volatility regime: ATR > 1.5 * ATR MA(50)
    vol_regime = atr_14 > (1.5 * atr_ma_50)
    
    # Get 12h data for Donchian channel calculation (prior bar OHLC)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from prior bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high(20) and low(20)
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 12h
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Calculate volume MA(20) for confirmation (using 12h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20, 20)  # Need enough bars for ATR, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_regime_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above Donchian high AND high volatility regime AND volume confirmed
            if curr_close > donch_high_aligned[i] and vol_regime_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND high volatility regime AND volume confirmed
            elif curr_close < donch_low_aligned[i] and vol_regime_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below Donchian low (reversion to mean)
            if curr_close < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above Donchian high (reversion to mean)
            if curr_close > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATR_VolRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0