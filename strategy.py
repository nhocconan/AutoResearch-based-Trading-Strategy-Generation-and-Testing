#!/usr/bin/env python3
"""
4h_1d_RSI_Divergence_With_Volume_Confirmation
Hypothesis: On 4h timeframe, identify bullish and bearish RSI divergences (price makes higher low/lower high while RSI makes lower low/higher high) combined with volume confirmation (current volume > 1.5x 20-period average) to capture reversals. Works in both bull and bear markets as it identifies exhaustion points. Uses 1d RSI for smoother signal and avoids overtrading by requiring confluence of price action, momentum, and volume.
"""

name = "4h_1d_RSI_Divergence_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for RSI calculation (smoother, less noise)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d RSI (14 period) ---
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:14] = np.nan  # Not enough data for proper RSI
    
    # Align 1d RSI to 4h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # --- 4h Volume Average for confirmation ---
    vol_avg_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 30  # for RSI and volume average
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_avg_4h[i])):
            if position != 0:
                # Simple stoploss: 2x ATR from entry (using 4h range as proxy)
                atr_est = np.abs(high_4h[i] - low_4h[i])
                if position == 1 and close_4h[i] <= entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_4h[i] >= entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 4h average
        vol_confirm = volume_4h[i] > 1.5 * vol_avg_4h[i]
        
        if position == 0 and vol_confirm:
            # Look for RSI divergences (need at least 3 bars back)
            if i >= 3:
                # Bullish divergence: price makes higher low, RSI makes lower low
                if (low_4h[i] > low_4h[i-2] and 
                    rsi_1d_aligned[i] < rsi_1d_aligned[i-2] and
                    rsi_1d_aligned[i] < 40):  # Oversold condition
                    signals[i] = 0.25  # long
                    position = 1
                    entry_price = close_4h[i]
                # Bearish divergence: price makes lower high, RSI makes higher high
                elif (high_4h[i] < high_4h[i-2] and 
                      rsi_1d_aligned[i] > rsi_1d_aligned[i-2] and
                      rsi_1d_aligned[i] > 60):  # Overbought condition
                    signals[i] = -0.25  # short
                    position = -1
                    entry_price = close_4h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position: exit on bearish divergence or overbought RSI
                if (rsi_1d_aligned[i] > 70 or  # Overbought exit
                    (i >= 3 and high_4h[i] < high_4h[i-2] and 
                     rsi_1d_aligned[i] > rsi_1d_aligned[i-2])):  # Bearish divergence
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short position: exit on bullish divergence or oversold RSI
                if (rsi_1d_aligned[i] < 30 or  # Oversold exit
                    (i >= 3 and low_4h[i] > low_4h[i-2] and 
                     rsi_1d_aligned[i] < rsi_1d_aligned[i-2])):  # Bullish divergence
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals