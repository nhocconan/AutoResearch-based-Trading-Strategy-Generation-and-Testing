#!/usr/bin/env python3
"""
6h_1d_RangeBreakout_RSI_Confirmation
Hypothesis: On 6h timeframe, trade breakouts from daily ranges with RSI confirmation.
Long when price breaks above previous day's high with RSI(14) > 50 (momentum confirmation).
Short when price breaks below previous day's low with RSI(14) < 50.
Exit when price returns to previous day's close (mean reversion to daily equilibrium).
Works in bull markets by buying strength and in bear markets by selling weakness.
Uses daily range as dynamic support/resistance and RSI to avoid counter-trend breaks.
Target: 15-30 trades/year per symbol with controlled risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for range calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range (levels based on prior day)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Align to 6h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # RSI(14) on 6h close
    close_series = prices['close']
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral before warmup
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if daily levels not ready
        if (np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(prev_close_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: break above prev day high with bullish momentum
            if price > prev_high_aligned[i] and rsi_val > 50:
                signals[i] = 0.25
                position = 1
            # Short: break below prev day low with bearish momentum
            elif price < prev_low_aligned[i] and rsi_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to previous day's close (mean reversion)
            if price <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to previous day's close
            if price >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_RangeBreakout_RSI_Confirmation"
timeframe = "6h"
leverage = 1.0