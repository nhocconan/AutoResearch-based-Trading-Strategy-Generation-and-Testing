#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume spike confirmation.
Long when price breaks above Donchian upper band AND 1d ATR(14) > 1d ATR(50) AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band AND 1d ATR(14) > 1d ATR(50) AND volume > 2.0x 20-period average.
Exit when price retraces to Donchian midpoint or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee churn while maintaining profit potential.
Donchian channels provide clear trend-following structure with proven efficacy across market regimes.
ATR regime filter ensures we only trade during sufficient volatility regimes, avoiding choppy markets.
Volume spike confirms institutional participation in breakouts.
Works in bull markets (breakouts with volume in uptrend) and bear markets (breakdowns with volume in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # True Range calculation for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # Calculate Donchian(20) channels from 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2.0
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation (using 4h data)
    tr1_4h = np.abs(high - low)
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr1_4h[0] = 0
    tr2_4h[0] = 0
    tr3_4h[0] = 0
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # ATR50 needs 50, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr14_1d_aligned[i]) or np.isnan(atr50_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_4h_val = atr_4h[i]
        atr14_val = atr14_1d_aligned[i]
        atr50_val = atr50_1d_aligned[i]
        upper_band = highest_20[i]
        lower_band = lowest_20[i]
        midpoint = donchian_mid[i]
        
        if position == 0:
            # Volatility regime filter: only trade when 1d ATR(14) > ATR(50) (increasing volatility)
            vol_regime = atr14_val > atr50_val
            
            # Long: Break above Donchian upper band AND volume spike AND volatility regime
            if close[i] > upper_band and volume[i] > 2.0 * vol_ma_val and vol_regime:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower band AND volume spike AND volatility regime
            elif close[i] < lower_band and volume[i] > 2.0 * vol_ma_val and vol_regime:
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
            if position == 1 and close[i] <= midpoint:
                exit_signal = True
            elif position == -1 and close[i] >= midpoint:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_4h_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_4h_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dATRRegime_VolumeSpike_MidpointExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0