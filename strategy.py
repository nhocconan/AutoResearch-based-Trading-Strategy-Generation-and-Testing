#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d ATR(14) for regime filter (low volatility when ATR < 20-period ATR mean, high volatility when ATR > 20-period ATR mean).
- Entry: Price breaks above/below 12h Donchian(20) levels with volume > 2.0 * 20-period volume MA and ATR regime filter aligned.
- Exit: ATR-based stoploss (2.0 * ATR(14)) or Donchian level reversal (touch opposite Donchian level).
- Signal size: 0.25 discrete to balance capture and fee control.
Designed to work in both bull and bear markets by using ATR regime to filter volatility and Donchian breakouts for momentum capture.
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
    
    # Get 12h data for Donchian levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20) levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper/lower using previous 20 bars
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR(20) mean for regime comparison
    atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR regime: low volatility when ATR < ATR_MA, high volatility when ATR > ATR_MA
    atr_regime_low = atr_1d < atr_ma_20
    atr_regime_high = atr_1d > atr_ma_20
    
    atr_regime_low_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_low.astype(float))
    atr_regime_high_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_high.astype(float))
    
    # Calculate 12h volume MA(20) for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate 12h ATR(14) for stoploss
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr2_12h[0] = 0
    tr3_12h[0] = 0
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_regime_low_aligned[i]) or np.isnan(atr_regime_high_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(atr_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold) and ATR regime filter
            vol_confirmed = curr_volume > 2.0 * vol_ma_12h_aligned[i]
            
            # Determine ATR regime: prefer low volatility regime for breakouts
            regime_low = atr_regime_low_aligned[i] > 0.5
            regime_high = atr_regime_high_aligned[i] > 0.5
            
            # Long: price breaks above Donchian high AND low volatility regime AND volume confirmed
            if curr_high > donchian_high_aligned[i] and regime_low and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below Donchian low AND high volatility regime AND volume confirmed
            elif curr_low < donchian_low_aligned[i] and regime_high and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or price breaks below Donchian low (reversal signal)
            stop_loss = entry_price - 2.0 * atr_12h_aligned[i]
            if curr_low < stop_loss or curr_low < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or price breaks above Donchian high (reversal signal)
            stop_loss = entry_price + 2.0 * atr_12h_aligned[i]
            if curr_high > stop_loss or curr_high > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dATRRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0