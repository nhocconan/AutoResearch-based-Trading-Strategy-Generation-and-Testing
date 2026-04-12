#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_breakout_v27
# Uses daily Camarilla pivot levels (H4/L4) for mean reversion in ranging markets.
# Long when price closes below L4 and reverses upward with volume confirmation.
# Short when price closes above H4 and reverses downward with volume confirmation.
# Uses 4h RSI(14) to avoid overbought/oversold extremes and filter false signals.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in ranging markets (mean reversion) and avoids trending markets via RSI filter.

name = "4h_1d_camarilla_breakout_v27"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # Using previous day's values to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 4h timeframe (daily values update after daily bar closes)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 4h RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price closes below L4 and reverses upward (close > L4)
        # Only in non-overbought conditions (RSI < 70)
        if close[i] <= L4_aligned[i] and close[i] > L4_aligned[i] * 1.001 and rsi[i] < 70 and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price closes above H4 and reverses downward (close < H4)
        # Only in non-oversold conditions (RSI > 30)
        elif close[i] >= H4_aligned[i] and close[i] < H4_aligned[i] * 0.999 and rsi[i] > 30 and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to midpoint or opposite signal
        elif (close[i] >= (H4_aligned[i] + L4_aligned[i]) / 2 and position == 1) or \
             (close[i] <= (H4_aligned[i] + L4_aligned[i]) / 2 and position == -1):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals