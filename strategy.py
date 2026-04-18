#!/usr/bin/env python3
"""
1d_1W_Pivot_R1S1_Breakout_Volume_Momentum
Hypothesis: Use weekly pivot levels (R1/S1) on weekly timeframe with daily entry. 
Long when price breaks above weekly R1 with volume > 1.5x average and RSI > 50 (momentum).
Short when price breaks below weekly S1 with volume > 1.5x average and RSI < 50.
Weekly pivot provides structural levels that work in both trending and ranging markets.
Volume confirmation ensures institutional participation. RSI filter avoids counter-trend entries.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
Works in bull/bear via momentum filter and weekly structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous week's OHLC for weekly pivot calculation
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close[0] = close_1w[0]  # first week uses same week
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    
    # Weekly pivot point and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    
    # RSI(14) for momentum filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], 100).fillna(100).values
    
    # Align weekly data to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need enough for RSI and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above weekly R1 with volume and bullish momentum
            if close[i] > r1_aligned[i] and vol_confirm and rsi[i] > 50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1 with volume and bearish momentum
            elif close[i] < s1_aligned[i] and vol_confirm and rsi[i] < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below weekly R1 or momentum turns bearish
            if close[i] < r1_aligned[i] or rsi[i] < 50:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above weekly S1 or momentum turns bullish
            if close[i] > s1_aligned[i] or rsi[i] > 50:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Pivot_R1S1_Breakout_Volume_Momentum"
timeframe = "1d"
leverage = 1.0