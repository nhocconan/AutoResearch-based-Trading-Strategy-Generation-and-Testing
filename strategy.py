#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_keltner_channel_breakout
# Uses daily Keltner Channels (ATR-based) as dynamic support/resistance on 4h chart.
# Long when price closes above upper KC (EMA + 2*ATR) with volume confirmation.
# Short when price closes below lower KC (EMA - 2*ATR) with volume confirmation.
# Exits when price crosses the middle line (EMA).
# Keltner Channels adapt to volatility, reducing false breakouts in low vol regimes.
# Volume confirmation filters out weak breakouts. Designed for low trade frequency.

name = "4h_1d_keltner_channel_breakout"
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
    
    # Get daily data for Keltner Channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA and ATR for Keltner Channels
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period EMA (middle line)
    ema_20 = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range and ATR (20-period)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bounds
    kc_upper = ema_20 + (2.0 * atr)
    kc_lower = ema_20 - (2.0 * atr)
    kc_middle = ema_20  # EMA as middle line
    
    # Align daily Keltner levels to 4h timeframe
    kc_upper_aligned = align_htf_to_ltf(prices, df_1d, kc_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_1d, kc_lower)
    kc_middle_aligned = align_htf_to_ltf(prices, df_1d, kc_middle)
    
    # Volume confirmation: volume > 1.3 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(kc_upper_aligned[i]) or np.isnan(kc_lower_aligned[i]) or np.isnan(kc_middle_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price closes above upper KC
        if close[i] > kc_upper_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price closes below lower KC
        elif close[i] < kc_lower_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses middle line (EMA)
        elif position == 1 and close[i] <= kc_middle_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= kc_middle_aligned[i]:
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