#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_v2
# Hypothesis: Breakout above/below 1-day Camarilla pivot levels with weekly trend filter.
# Long when price breaks above H3 with weekly uptrend (price > weekly EMA50).
# Short when price breaks below L3 with weekly downtrend (price < weekly EMA50).
# Exit when price returns to weekly EMA50.
# Weekly trend filter prevents counter-trend trades in choppy markets.
# Target: 10-25 trades/year (40-100 total over 4 years) with strict entry conditions.
# Works in both bull and bear markets due to trend filter reducing false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate True Range for ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # ATR(14) for volatility filter
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]  # Wilder's smoothing
    
    # Previous day's data for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_h3 = prev_close + range_val * 1.1 / 4
    camarilla_l3 = prev_close - range_val * 1.1 / 4
    
    # Load weekly data ONCE before loop for trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA(50)
    close_1w = df_1w['close'].values
    ema_1w = np.zeros_like(close_1w, dtype=float)
    ema_1w[0] = close_1w[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_1w)):
        ema_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_1w[i-1]
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start after first bar (need previous day)
        # Skip if any required data is invalid
        if np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or np.isnan(atr[i]) or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility (>5% of price)
        vol_filter = atr[i] < 0.05 * close[i]
        
        if position == 1:  # Long position
            # Exit: price returns to weekly EMA (mean reversion in trend)
            if close[i] <= ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly EMA
            if close[i] >= ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above H3 with volatility filter and weekly uptrend
            if close[i] > camarilla_h3[i] and vol_filter and close[i] > ema_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below L3 with volatility filter and weekly downtrend
            elif close[i] < camarilla_l3[i] and vol_filter and close[i] < ema_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals