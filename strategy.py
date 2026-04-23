#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d ATR(14) > 20-period average ATR (high volatility regime) AND volume > 1.5x 20-period average volume.
Short when price breaks below Donchian lower band AND 1d ATR(14) > 20-period average ATR AND volume > 1.5x 20-period average volume.
Exit when price retraces to Donchian midpoint OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~20-35 trades/year on 4h timeframe.
Donchian channels provide clear structural breakouts; volatility regime filter avoids low-chop environments; volume confirmation adds conviction.
Works in bull (breakouts with expansion) and bear (breakdowns with expansion) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = np.abs(high_1d - low_1d)
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # 20-period average of 1d ATR for regime threshold
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # Donchian channels (20-period) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 4h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian needs 20, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma_1d_aligned[i]) or
            np.isnan(atr_1d[-1] if len(atr_1d) > 0 else np.nan)):  # atr_1d[-1] for current 1d ATR
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        atr_ma_1d_val = atr_ma_1d_aligned[i]
        upper_band = highest_20[i]
        lower_band = lowest_20[i]
        mid_val = donchian_mid[i]
        
        # Current 1d ATR (use last value of 1d ATR array, aligned to current 4h bar)
        # Since we aligned the MA, we need the current 1d ATR value
        # Get the index of the 1d bar that corresponds to current 4h bar
        # Simpler: use the aligned ATR_1d (not MA) for regime check
        # Re-align the raw 1d ATR
        atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
        atr_1d_val = atr_1d_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian upper band AND high volatility regime (1d ATR > its MA) AND volume spike (1.5x avg)
            if close[i] > upper_band and atr_1d_val > atr_ma_1d_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower band AND high volatility regime AND volume spike
            elif close[i] < lower_band and atr_1d_val > atr_ma_1d_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Donchian midpoint
            if position == 1 and close[i] <= mid_val:
                exit_signal = True
            elif position == -1 and close[i] >= mid_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dATR_Regime_VolumeConfirmation_MidExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0