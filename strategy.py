#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_channel_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and Keltner calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily trend: EMA(21)
    ema_21 = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21)
    
    # Previous daily bar's data for Keltner Channels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate daily ATR for Keltner Channels
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - np.roll(prev_close, 1))
    tr3 = np.abs(prev_low - np.roll(prev_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan  # First value has no previous close
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channels: EMA(20) ± 2 * ATR
    keltner_middle = pd.Series(prev_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = keltner_middle + 2 * atr
    keltner_lower = keltner_middle - 2 * atr
    
    # Map daily Keltner levels to 4h bars
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower)
    keltner_middle_aligned = align_htf_to_ltf(prices, df_1d, keltner_middle)
    
    # Volume confirmation: current 4h volume > 20-period average of daily volume
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
    vol_ma = pd.Series(vol_1d_aligned).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above upper Keltner with volume and above daily EMA
        long_signal = (close[i] > keltner_upper_aligned[i] and volume_filter[i] and close[i] > ema_21_aligned[i])
        
        # Short: price breaks below lower Keltner with volume and below daily EMA
        short_signal = (close[i] < keltner_lower_aligned[i] and volume_filter[i] and close[i] < ema_21_aligned[i])
        
        # Exit: price returns to middle Keltner line
        exit_long = (position == 1 and close[i] < keltner_middle_aligned[i])
        exit_short = (position == -1 and close[i] > keltner_middle_aligned[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals