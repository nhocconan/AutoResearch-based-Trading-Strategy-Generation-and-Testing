#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Williams %R extremes with 1d EMA50 trend filter and volume confirmation.
Long when 1d Williams %R crosses above -80 (oversold recovery) AND price > 1d EMA50 AND volume > 1.5x 20-period average.
Short when 1d Williams %R crosses below -20 (overbought deterioration) AND price < 1d EMA50 AND volume > 1.5x 20-period average.
Exit when Williams %R returns to -50 (mean reversion) or ATR trailing stop hit (2.5*ATR from extreme).
Williams %R captures momentum exhaustion in both bull and bear markets, while EMA50 filter ensures trend alignment.
Designed for 6h timeframe targeting ~20 trades/year per symbol (80 total over 4 years).
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
    
    # Calculate 1d Williams %R and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    extreme_since_entry = 0.0  # highest for long, lowest for short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20)  # Williams needs 14, EMA needs 50, vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        wr_val = williams_r_aligned[i]
        ema_50_val = ema_50_aligned[i]
        wr_prev = williams_r_aligned[i-1]
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold recovery) AND price > EMA50 AND volume spike
            if (wr_prev <= -80 and wr_val > -80 and price > ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
                extreme_since_entry = price
            # Short: Williams %R crosses below -20 (overbought deterioration) AND price < EMA50 AND volume spike
            elif (wr_prev >= -20 and wr_val < -20 and price < ema_50_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
                extreme_since_entry = price
        else:
            # Update extreme since entry for trailing stop
            if position == 1:
                extreme_since_entry = max(extreme_since_entry, price)
            elif position == -1:
                extreme_since_entry = min(extreme_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R returns to -50 (mean reversion)
            if position == 1 and wr_val >= -50:
                exit_signal = True
            elif position == -1 and wr_val <= -50:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from extreme since entry
            if position == 1 and price < extreme_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > extreme_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                extreme_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Extremes_1dEMA50_Trend_VolumeConfirmation_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0