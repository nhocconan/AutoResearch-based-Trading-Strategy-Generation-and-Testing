#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_camarilla_momentum_v1
# Combines 4h/1d Camarilla pivot levels with 1h momentum (RSI) for entry timing.
# Uses 4h for trend direction (price above/below 4h VWAP), 1d Camarilla for support/resistance,
# and 1h RSI pullback for entry. Reduces trades by requiring confluence of trend, level, and momentum.
# Target: 15-35 trades/year per symbol.
name = "1h_4h_1d_camarilla_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (precomputed)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # 4h trend: VWAP
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    vwap_4h = (typical_price_4h * df_4h['volume']).cumsum() / df_4h['volume'].cumsum()
    vwap_4h_values = vwap_4h.values
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h_values)
    
    # 1d Camarilla levels (from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    range_prev = high_prev - low_prev
    camarilla_h3 = close_prev + range_prev * 1.1 / 4
    camarilla_l3 = close_prev - range_prev * 1.1 / 4
    h3_level = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_level = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not session_mask[i]:
            # Outside session: flatten
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if levels not ready
        if np.isnan(vwap_4h_aligned[i]) or np.isnan(h3_level[i]) or np.isnan(l3_level[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # Long: uptrend (above 4h VWAP), pullback to L3, RSI < 40
        if close[i] > vwap_4h_aligned[i] and close[i] <= l3_level[i] and rsi[i] < 40 and position != 1:
            position = 1
            signals[i] = 0.20
        # Short: downtrend (below 4h VWAP), pullback to H3, RSI > 60
        elif close[i] < vwap_4h_aligned[i] and close[i] >= h3_level[i] and rsi[i] > 60 and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: trend reversal
        elif (position == 1 and close[i] < vwap_4h_aligned[i]) or \
             (position == -1 and close[i] > vwap_4h_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals