#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1d Williams %R extreme readings with volume confirmation and 1d EMA34 trend filter.
Long when 1d Williams %R < -80 (oversold) AND price > 1d EMA34 (bullish trend) AND volume > 1.5x 20-period average.
Short when 1d Williams %R > -20 (overbought) AND price < 1d EMA34 (bearish trend) AND volume > 1.5x 20-period average.
Exit when Williams %R returns to -50 level or ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to balance return and drawdown.
Designed for 6h timeframe to target 12-37 trades/year per symbol (50-150 total over 4 years).
Williams %R identifies exhaustion points; volume confirmation filters weak moves; EMA34 ensures trend alignment.
Works in both bull and bear markets by fading extremes only when aligned with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams %R and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # EMA34 on 1d close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        williams_r_val = williams_r_aligned[i]
        ema_34_val = ema_34_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price above 1d EMA34 (bullish trend) AND volume spike
            if (williams_r_val < -80 and price > ema_34_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Williams %R overbought (> -20) AND price below 1d EMA34 (bearish trend) AND volume spike
            elif (williams_r_val > -20 and price < ema_34_val and volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Williams %R returns to -50 level (mean reversion)
            if position == 1 and williams_r_val >= -50:
                exit_signal = True
            elif position == -1 and williams_r_val <= -50:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_EMA34_VolumeConfirmation_ATRStop"
timeframe = "6h"
leverage = 1.0