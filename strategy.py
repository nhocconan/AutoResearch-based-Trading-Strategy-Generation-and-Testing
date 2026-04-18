#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for direction and 1d RSI for overbought/oversold filters.
# Supertrend captures trend direction with built-in ATR-based stop.
# 1d RSI < 30 for longs, > 70 for shorts to trade against extreme sentiment on higher timeframe.
# Volume spike (>1.5x 20-period average) confirms conviction at entry.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.
name = "1h_Supertrend4h_1dRSI_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Supertrend (trend direction)
    df_4h = get_htf_data(prices, '4h')
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Supertrend on 4h data
    # ATR calculation
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_period = 10
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < atr_period:
            atr[i] = np.nan
        elif i == atr_period:
            atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    factor = 3.0
    hl2 = (high_4h + low_4h) / 2
    upperband = hl2 + (factor * atr)
    lowerband = hl2 - (factor * atr)
    
    # Initialize Supertrend arrays
    supetrend = np.zeros_like(close_4h)
    supetrend_up = np.zeros_like(close_4h)
    supetrend_down = np.zeros_like(close_4h)
    
    # Set initial values
    supetrend[0] = 1  # start with uptrend
    supetrend_up[0] = upperband[0]
    supetrend_down[0] = lowerband[0]
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > supetrend_up[i-1]:
            supetrend[i] = 1
        elif close_4h[i] < supetrend_down[i-1]:
            supetrend[i] = -1
        else:
            supetrend[i] = supetrend[i-1]
            
        if supetrend[i] == 1:
            supetrend_up[i] = max(upperband[i], supetrend_up[i-1])
            supetrend_down[i] = lowerband[i]
        else:
            supetrend_up[i] = upperband[i]
            supetrend_down[i] = min(lowerband[i], supetrend_down[i-1])
    
    # Align Supertrend to 1h timeframe
    supetrend_aligned = align_htf_to_ltf(prices, df_4h, supetrend)
    
    # Calculate 1d RSI for overbought/oversold filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # RSI smoothing (Wilder's smoothing)
    rsi_period = 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    # Initial average
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    # Wilder's smoothing
    for i in range(rsi_period+1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    # Calculate RSI
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supetrend_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from Supertrend
        uptrend = supetrend_aligned[i] == 1
        downtrend = supetrend_aligned[i] == -1
        
        if position == 0:
            # Long: uptrend AND RSI < 30 (oversold) AND volume spike
            if uptrend and rsi_1d_aligned[i] < 30 and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: downtrend AND RSI > 70 (overbought) AND volume spike
            elif downtrend and rsi_1d_aligned[i] > 70 and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend turns down OR RSI > 70 (overbought)
            if not uptrend or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend turns up OR RSI < 30 (oversold)
            if not downtrend or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals