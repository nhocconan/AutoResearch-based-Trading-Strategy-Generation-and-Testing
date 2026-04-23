#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d EMA50 trend filter and volume spike confirmation.
Long when price breaks above Donchian upper (20) AND 1d close > 1d EMA50 AND 1d volume > 2.0x 20-period average volume.
Short when price breaks below Donchian lower (20) AND 1d close < 1d EMA50 AND 1d volume > 2.0x 20-period average volume.
Exit when price retraces to Donchian midpoint OR ATR trailing stop (2.5*ATR from extreme).
Uses discrete position sizing (0.25) targeting ~20-35 trades/year on 4h timeframe.
Donchian channels provide objective structure that works in both bull and bear markets when combined with trend and volume filters.
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
    
    # Calculate 1d OHLC for Donchian and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 and Donchian
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Donchian channels (20-period) for 1d
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align Donchian levels to 4h (completed 1d bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for spike filter
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
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
    start_idx = max(50, 20)  # EMA50 needs 50, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_1d_val = vol_ma_1d_aligned[i]
        atr_val = atr[i]
        ema_val = ema_50_1d_aligned[i]
        upper_val = donchian_high_aligned[i]
        lower_val = donchian_low_aligned[i]
        mid_val = donchian_mid_aligned[i]
        
        # Current 1d values (use last value of 1d arrays, aligned to current 4h bar)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        vol_1d_val = vol_1d_aligned[i]
        close_1d_val = close_1d_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian upper AND bullish trend (close > EMA50) AND volume spike
            if close[i] > upper_val and close_1d_val > ema_val and vol_1d_val > 2.0 * vol_ma_1d_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Donchian lower AND bearish trend (close < EMA50) AND volume spike
            elif close[i] < lower_val and close_1d_val < ema_val and vol_1d_val > 2.0 * vol_ma_1d_val:
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
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA50_Trend_VolumeConfirmation_MidExit_ATRTrailingStop"
timeframe = "4h"
leverage = 1.0