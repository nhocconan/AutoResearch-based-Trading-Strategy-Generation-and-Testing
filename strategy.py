#!/usr/bin/env python3
"""
1d_1W_Momentum_Reversal
Hypothesis: Capture mean reversion after weekly momentum extremes.
Long when price closes below weekly low with RSI(14) < 30 and volume > 1.5x average.
Short when price closes above weekly high with RSI(14) > 70 and volume > 1.5x average.
Exit when RSI returns to neutral (40-60) or opposite extreme is hit.
Target: 10-20 trades/year per symbol (40-80 total over 4 years) to minimize fee drag.
Works in bull/bear via momentum exhaustion signals.
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
    
    # Get weekly data for momentum extremes
    df_1w = get_htf_data(prices, '1w')
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align weekly data to daily timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for RSI and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below weekly low, oversold RSI, with volume confirmation
            if (close[i] < low_1w_aligned[i] and 
                rsi_1w_aligned[i] < 30 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price above weekly high, overbought RSI, with volume confirmation
            elif (close[i] > high_1w_aligned[i] and 
                  rsi_1w_aligned[i] > 70 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral or hits opposite extreme
            if rsi_1w_aligned[i] > 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral or hits opposite extreme
            if rsi_1w_aligned[i] < 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1W_Momentum_Reversal"
timeframe = "1d"
leverage = 1.0