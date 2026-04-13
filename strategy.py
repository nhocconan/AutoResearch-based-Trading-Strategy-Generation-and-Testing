#!/usr/bin/env python3
"""
6h_1d_Pivot_Momentum_Reversal_v1
Hypothesis: Price reverses from daily pivot points (PP) with momentum confirmation.
In ranging markets, price respects daily pivot as support/resistance. In trending markets,
breakouts from pivot with momentum (RSI divergence) continue. Uses 60% position size
to manage drawdown. Works in both bull and bear by adapting to regime via RSI(6).
Target: 15-25 trades/year per symbol.
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
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points: PP = (H+L+C)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*PP - L, S1 = 2*PP - H
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # RSI(6) for momentum confirmation
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/6, adjust=False, min_periods=6).mean()
    avg_loss = loss.ewm(alpha=1/6, adjust=False, min_periods=6).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price near S1 with bullish momentum OR break above R1 with momentum
        near_support = close[i] <= s1_aligned[i] * 1.005  # within 0.5% of S1
        bullish_momentum = rsi[i] > 50 and rsi[i] > rsi[i-1]  # RSI > 50 and rising
        breakout_resistance = close[i] > r1_aligned[i] and rsi[i] > 60  # breakout with strong momentum
        
        # Short conditions: price near R1 with bearish momentum OR break below S1 with momentum
        near_resistance = close[i] >= r1_aligned[i] * 0.995  # within 0.5% of R1
        bearish_momentum = rsi[i] < 50 and rsi[i] < rsi[i-1]  # RSI < 50 and falling
        breakdown_support = close[i] < s1_aligned[i] and rsi[i] < 40  # breakdown with weak momentum
        
        if (near_support and bullish_momentum) or breakout_resistance:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        elif (near_resistance and bearish_momentum) or breakdown_support:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "6h_1d_Pivot_Momentum_Reversal_v1"
timeframe = "6h"
leverage = 1.0