#!/usr/bin/env python3
"""
Hypothesis: 6-hour Supertrend (ATR-based trend) combined with 12-hour RSI mean reversion and volume confirmation.
Long when price is above Supertrend (uptrend), RSI < 30 (oversold), and volume > 1.3x average.
Short when price is below Supertrend (downtrend), RSI > 70 (overbought), and volume > 1.3x average.
Exit when price crosses back through Supertrend or RSI returns to neutral (40-60).
Designed for low trade frequency (~15-30/year) to capture mean reversion within established trends.
Works in both bull and bear markets by aligning mean reversion with the dominant trend direction.
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
    
    # Load 12-hour data for RSI - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12-hour RSI (14-period)
    close_12h = df_12h['close'].values
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to alpha=1/14)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Supertrend on LTF (6h)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR
    atr = np.zeros_like(close)
    atr[atr_period-1] = np.mean(tr[0:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.zeros_like(close)
    supertrend[:] = np.nan
    direction = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    supertrend[atr_period-1] = upper_band[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, len(close)):
        if close[i] > supertrend[i-1]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            direction[i] = -1
    
    # Align HTF indicators to lower timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # Volume average (20-period) on lower timeframe
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(atr_period, 20) + 1, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(direction[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        rsi_val = rsi_aligned[i]
        supertrend_val = supertrend[i]
        direction_val = direction[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        
        if position == 0:
            # Long: price above Supertrend (uptrend), RSI oversold, volume confirmation
            if (close_val > supertrend_val and direction_val == 1 and
                rsi_val < 30 and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price below Supertrend (downtrend), RSI overbought, volume confirmation
            elif (close_val < supertrend_val and direction_val == -1 and
                  rsi_val > 70 and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Supertrend OR RSI returns to neutral (40-60)
                if close_val < supertrend_val or (rsi_val >= 40 and rsi_val <= 60):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Supertrend OR RSI returns to neutral (40-60)
                if close_val > supertrend_val or (rsi_val >= 40 and rsi_val <= 60):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Supertrend_12hRSI_Volume_MeanReversion"
timeframe = "6h"
leverage = 1.0