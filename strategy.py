#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly trend filter (1w close > 1w EMA50) and 1d volume spike confirmation.
Long when price breaks above 6h Donchian upper (20) AND 1w close > 1w EMA50 AND 1d volume > 2.0x 20-period average volume.
Short when price breaks below 6h Donchian lower (20) AND 1w close < 1w EMA50 AND 1d volume > 2.0x 20-period average volume.
Exit when price reaches 6h Donchian midpoint OR ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~12-37 trades/year on 6h timeframe.
Combines price structure (Donchian channels), trend filter (1w EMA50), and volume confirmation for robustness across bull/bear regimes.
Weekly trend ensures alignment with major market direction, reducing whipsaw in ranging markets.
Volume spike confirms institutional participation in breakouts.
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
    
    # Calculate 1w OHLC for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # 1w arrays for EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d OHLC for volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for volume MA
        return np.zeros(n)
    
    # 1d volume for spike filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 6h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_max_20
    donchian_lower = low_min_20
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # ATR(14) for 6h trailing stop calculation
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
    start_idx = max(20, 50)  # Donchian20 and EMA50 need 20 and 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma_20_1d_aligned[i]
        atr_val = atr[i]
        ema_val = ema_50_1w_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        mid = donchian_mid[i]
        
        # Current 1d close value (use last value of 1d arrays, aligned to current 6h bar)
        close_1d = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        close_1d_val = close_1d_aligned[i]
        vol_1d_val = volume_1d[i]  # current 1d volume (already completed bar)
        
        if position == 0:
            # Long: Break above Donchian upper AND bullish weekly trend (close > EMA50) AND volume spike
            if close[i] > upper and close_1d_val > ema_val and vol_1d_val > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower AND bearish weekly trend (close < EMA50) AND volume spike
            elif close[i] < lower and close_1d_val < ema_val and vol_1d_val > 2.0 * vol_ma_val:
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
            
            # Primary exit: Price reaches 6h Donchian midpoint
            if position == 1 and close[i] <= mid:
                exit_signal = True
            elif position == -1 and close[i] >= mid:
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

name = "6H_Donchian20_Breakout_1wEMA50_Trend_1dVolumeSpike_MidExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0