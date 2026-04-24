#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility regime filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ATR(14) to measure volatility regime - high volatility (ATR > ATR MA(50)) enables breakout trading.
- Donchian channels: 20-period high/low from prior 4h bars for breakout detection.
- Entry: Long when price breaks above prior 20-bar Donchian high AND 1d ATR > 1.2 * ATR MA(50) AND volume > 1.5 * volume MA(20).
         Short when price breaks below prior 20-bar Donchian low AND 1d ATR > 1.2 * ATR MA(50) AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below prior 20-bar Donchian low,
        exit short when price crosses above prior 20-bar Donchian high.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Uses volatility regime filter to avoid whipsaws in low-volatility markets and capture momentum in high-volatility periods.
Works in both bull and bear markets via volatility-based regime filtering and mean-reversion exits.
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
    
    # Get 1d data for ATR-based volatility regime filter
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
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # ATR(14) - exponential moving average of TR
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # ATR MA(50) for regime filter
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF indicators to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Calculate Donchian(20) channels from 4h data (prior 20 bars only)
    # We need to look back 20 bars, so we'll calculate it manually to avoid look-ahead
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        # Prior 20 bars: i-20 to i-1 (excluding current bar i)
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need enough bars for ATR MA(50) and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volatility regime and volume confirmation
            vol_regime = atr_1d_aligned[i] > 1.2 * atr_ma_1d_aligned[i]
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above prior 20-bar Donchian high AND high volatility regime AND volume confirmed
            if curr_close > donchian_high[i] and vol_regime and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 20-bar Donchian low AND high volatility regime AND volume confirmed
            elif curr_close < donchian_low[i] and vol_regime and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior 20-bar Donchian low (mean reversion)
            if curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior 20-bar Donchian high (mean reversion)
            if curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_VolRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0