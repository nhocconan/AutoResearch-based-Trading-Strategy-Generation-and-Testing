#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ATR(14) for volatility regime (high volatility when ATR > ATR MA(50)).
- Donchian channel: 20-period high/low from 4h data for breakout detection.
- Entry: Long when price breaks above Donchian(20) high AND 1d ATR > 1.2 * ATR MA(50) AND volume > 1.5 * volume MA(20).
         Short when price breaks below Donchian(20) low AND 1d ATR > 1.2 * ATR MA(50) AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below Donchian(20) low,
        exit short when price crosses above Donchian(20) high.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to capture volatile breakouts in both bull and bear markets via volatility filter.
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
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # ATR MA(50) for regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Calculate Donchian(20) channel from 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 50, 20)  # Need enough bars for all indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume and volatility confirmation
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            vol_regime = atr_1d_aligned[i] > 1.2 * atr_ma_1d_aligned[i]
            
            # Long: Price breaks above Donchian high AND volatility regime AND volume confirmed
            if curr_close > donchian_high[i] and vol_regime and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND volatility regime AND volume confirmed
            elif curr_close < donchian_low[i] and vol_regime and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below Donchian low (mean reversion)
            if curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above Donchian high (mean reversion)
            if curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_Regime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0