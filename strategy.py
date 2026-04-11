#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_v1
# Strategy: 4h Camarilla pivot breakout with volume confirmation and RSI filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels act as strong support/resistance on daily timeframe.
# Breakouts above H3 or below L3 with volume surge and RSI momentum capture institutional moves.
# Works in bull markets via breakout continuation and bear markets via breakdowns.
# Low trade frequency (~20-50/year) minimizes fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    # H4, H3, H2, H1, L1, L2, L3, L4
    # Using formula: Close + (High - Low) * multiplier
    # and Close - (High - Low) * multiplier
    # We'll use H3 and L3 as primary breakout levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    h3 = prev_close + rang * 1.1 / 6
    l3 = prev_close - rang * 1.1 / 6
    
    # Align to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current 4h volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    # RSI filter: avoid overbought/oversold conditions
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI range: 30-70 to avoid extremes
    rsi_filter = (rsi > 30) & (rsi < 70)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        breakout_up = close[i] > h3_aligned[i]
        breakdown_down = close[i] < l3_aligned[i]
        
        # Entry conditions with filters
        # Long: breakout above H3 + volume confirmation + RSI not overbought
        if breakout_up and vol_confirm[i] and rsi_filter[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: breakdown below L3 + volume confirmation + RSI not oversold
        elif breakdown_down and vol_confirm[i] and rsi_filter[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price returns to pivot level (mean reversion)
        elif position == 1 and close[i] < prev_close[i//16 if i>=16 else 0]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > prev_close[i//16 if i>=16 else 0]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals