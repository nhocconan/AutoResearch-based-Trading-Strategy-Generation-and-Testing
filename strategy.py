#!/usr/bin/env python3
# 4h_1D_Momentum_Reversal_With_Volume
# Hypothesis: 4h momentum reversals at 1-day highs/lows with volume confirmation.
# Long when: price breaks above 1-day high AND volume > 1.5x 20-bar avg AND RSI(14) < 70 (avoid overextended)
# Short when: price breaks below 1-day low AND volume > 1.5x 20-bar avg AND RSI(14) > 30 (avoid overextended)
# Exit when: price crosses 1-day midpoint OR RSI reaches opposite extreme (RSI>70 for long, RSI<30 for short)
# Uses 1-day high/low as dynamic support/resistance. Works in bull by buying breakouts above prior day high;
# works in bear by selling breakdowns below prior day low. Volume filters false breakouts. RSI prevents chasing.
# Target: 25-40 trades/year (100-160 total over 4 years) to avoid fee drag.

name = "4h_1D_Momentum_Reversal_With_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1-day high, low, close ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # --- 1-day midpoint (for exit) ---
    midpoint_1d = (high_1d + low_1d) / 2
    
    # --- 4h RSI(14) ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[0:15])  # 14 periods + current
            avg_loss[i] = np.mean(loss[0:15])
        else:
            avg_gain[i] = (gain[i] + 13 * avg_gain[i-1]) / 14
            avg_loss[i] = (loss[i] + 13 * avg_loss[i-1]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # --- 4h volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 4h
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    midpoint_1d_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(1d data needs 1 bar, RSI14, vol MA20)
    start_idx = max(1, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(high_1d_aligned[i]) or
            np.isnan(low_1d_aligned[i]) or
            np.isnan(midpoint_1d_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            # Long: break above 1-day high with volume, not overextended
            if close[i] > high_1d_aligned[i] and vol_spike and rsi[i] < 70:
                signals[i] = 0.25
                position = 1
            # Short: break below 1-day low with volume, not overextended
            elif close[i] < low_1d_aligned[i] and vol_spike and rsi[i] > 30:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses 1-day midpoint OR RSI overbought
                if close[i] < midpoint_1d_aligned[i] or rsi[i] >= 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses 1-day midpoint OR RSI oversold
                if close[i] > midpoint_1d_aligned[i] or rsi[i] <= 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals