#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
Long when price breaks above Donchian upper (20) AND ATR(14) > ATR(50) (high volatility regime) AND volume > 1.5x 20-period average.
Short when price breaks below Donchian lower (20) AND ATR(14) > ATR(50) AND volume > 1.5x 20-period average.
Exit when price retraces to Donchian midpoint OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~25 trades/year on 12h timeframe.
Donchian channels provide clear structure, ATR regime filter avoids low-volatility whipsaws, volume confirmation ensures conviction.
Works in both bull (breakouts) and bear (breakdowns) markets by capturing expansion moves.
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
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR values to 12h timeframe
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # Calculate Donchian channels (20-period) from 1d
    donch_hi_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_lo_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid_1d = (donch_hi_1d + donch_lo_1d) / 2.0
    
    # Align Donchian levels to 12h timeframe
    donch_hi_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_hi_1d)
    donch_lo_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_lo_1d)
    donch_mid_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation (using 12h data)
    tr1_12h = np.abs(high - low)
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr1_12h[0] = 0
    tr2_12h[0] = 0
    tr3_12h[0] = 0
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14, 20)  # ATR50 needs 50, Donchian needs 20, ATR needs 14, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr14_1d_aligned[i]) or np.isnan(atr50_1d_aligned[i]) or
            np.isnan(donch_hi_1d_aligned[i]) or np.isnan(donch_lo_1d_aligned[i]) or np.isnan(donch_mid_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr_12h[i]
        atr14_val = atr14_1d_aligned[i]
        atr50_val = atr50_1d_aligned[i]
        donch_hi = donch_hi_1d_aligned[i]
        donch_lo = donch_lo_1d_aligned[i]
        donch_mid = donch_mid_1d_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian upper AND high volatility regime (ATR14 > ATR50) AND volume spike
            if close[i] > donch_hi and atr14_val > atr50_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower AND high volatility regime AND volume spike
            elif close[i] < donch_lo and atr14_val > atr50_val and volume[i] > 1.5 * vol_ma_val:
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
            if position == 1 and close[i] <= donch_mid:
                exit_signal = True
            elif position == -1 and close[i] >= donch_mid:
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

name = "12H_Donchian20_Breakout_1dATR_Regime_VolumeConfirmation_MidExit_ATRTrailingStop"
timeframe = "12h"
leverage = 1.0