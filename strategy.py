#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d ATR(30) to define volatility regime (low ATR = range, high ATR = breakout-prone).
- Entry: Long when price breaks above Donchian upper channel AND 1d ATR > its 20-period MA AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below Donchian lower channel AND 1d ATR > its 20-period MA AND volume > 2.0 * 4h volume MA(20).
- Exit: ATR trailing stop (2.5 * ATR(14)) or opposite Donchian breakout.
- Signal size: 0.25 discrete to minimize fee churn.
- Uses volatility expansion (high ATR) to catch strong moves while avoiding choppy low-volatility periods.
- Volume spike requirement (>2x MA) ensures conviction, reducing false breakouts.
- Works in bull (breakouts with rising vol) and bear (panic spikes with high ATR).
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
    
    # Get 4h data for Donchian calculation and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(30) for volatility regime
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=30, min_periods=30).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align ATR and its MA to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate 4h volume MA(20) for confirmation
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Calculate 4h ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 20, 14, 1)  # ATR regime needs 30+20, Donchian needs 20, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(atr_ma_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volatility regime: 1d ATR > its 20-period MA (expanding volatility)
        vol_regime = atr_1d_aligned[i] > atr_ma_1d_aligned[i]
        
        # Volume confirmation: 2.0x threshold (strict to reduce trades)
        vol_confirm = curr_volume > 2.0 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above Donchian high in high vol regime with volume spike
            if curr_close > donchian_high_aligned[i] and vol_regime and vol_confirm:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below Donchian low in high vol regime with volume spike
            elif curr_close < donchian_low_aligned[i] and vol_regime and vol_confirm:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite breakout (below Donchian low)
            if curr_low <= highest_since_entry - 2.5 * atr[i] or curr_close < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite breakout (above Donchian high)
            if curr_high >= lowest_since_entry + 2.5 * atr[i] or curr_close > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRRegime_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0