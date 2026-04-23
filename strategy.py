#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation.
Long when price breaks above upper BB AND 1d EMA50 up AND volume > 1.5x 20-period average.
Short when price breaks below lower BB AND 1d EMA50 down AND volume > 1.5x 20-period average.
Exit when price reverts to middle BB (20 SMA) OR ATR trailing stop (2.5*ATR from extreme).
Bollinger squeeze captures low volatility breakouts; 1d EMA50 filters higher timeframe trend;
volume confirms institutional participation. Works in both bull and bear markets by
trading breakouts in direction of higher timeframe trend.
Target: ~15-25 trades/year on 6h timeframe with discrete sizing 0.25.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2.0
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_bb + bb_std * std_bb
    lower_bb = sma_bb - bb_std * std_bb
    middle_bb = sma_bb  # 20 SMA for exit
    
    # 6h volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 6h trailing stop
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
    start_idx = max(bb_period, 50)  # BB20, vol_ma20, ema_50_1d
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(sma_bb[i]) or np.isnan(std_bb[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or
            np.isnan(middle_bb[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_50_1d_aligned[i]
        upper = upper_bb[i]
        lower = lower_bb[i]
        middle = middle_bb[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        # Determine 1d trend direction (rising/falling EMA50)
        if i > start_idx:
            prev_ema = ema_50_1d_aligned[i-1]
            ema_rising = ema_trend > prev_ema
            ema_falling = ema_trend < prev_ema
        else:
            ema_rising = True  # neutral on first bar
            ema_falling = False
        
        if position == 0:
            # Long: price breaks above upper BB AND 1d EMA50 rising AND volume confirmation
            if price > upper and ema_rising and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: price breaks below lower BB AND 1d EMA50 falling AND volume confirmation
            elif price < lower and ema_falling and volume[i] > 1.5 * vol_ma_val:
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
            
            # Primary exit: price reverts to middle BB (20 SMA)
            if position == 1 and price < middle:
                exit_signal = True
            elif position == -1 and price > middle:
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

name = "6H_Bollinger_Squeeze_Breakout_1dEMA50_Trend_VolumeConfirmation_MiddleExit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0