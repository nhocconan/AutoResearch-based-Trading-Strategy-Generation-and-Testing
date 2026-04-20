#!/usr/bin/env python3
"""
6h_1d_Pivot_Breakout_Pullback_With_Momentum
Hypothesis: In trending markets, price breaks above/below key daily pivot levels (R1/S1) and pulls back to the level with momentum confirmation.
In ranging markets, price respects pivot levels as support/resistance, allowing mean-reversion entries.
Uses 6h for entry timing and 1d pivots for structure, with momentum filter to avoid false breakouts.
Target: 20-40 trades/year per symbol (80-160 total over 4 years).
Works in bull/bear: momentum filter adapts to trend direction, pivot levels provide structure in all regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_Breakout_Pullback_With_Momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: Calculate daily pivot points (standard) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and support/resistance levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    
    # Align pivot levels to 6h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # === 6h: Price, volume, and momentum ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Momentum: RSI(14) to avoid overextended entries
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current volume vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(rsi_values[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ratio_val = vol_ratio[i]
        rsi_val = rsi_values[i]
        
        # Entry conditions: volume confirmation + momentum filter
        vol_ok = vol_ratio_val > 1.2
        # Momentum: not overbought/oversold for mean reversion, not weak for breakout
        mom_ok_long = rsi_val < 70  # Not overbought
        mom_ok_short = rsi_val > 30  # Not oversold
        
        if position == 0:
            # Long: price breaks above R1 and pulls back to R1 with momentum
            if (price > r1_aligned[i] and  # Broke above R1
                low[i] <= r1_aligned[i] and  # Pulled back to touch R1
                vol_ok and mom_ok_long):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 and pulls back to S1 with momentum
            elif (price < s1_aligned[i] and  # Broke below S1
                  high[i] >= s1_aligned[i] and  # Pulled back to touch S1
                  vol_ok and mom_ok_short):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (trend reversal) or overextended RSI
            if price < s1_aligned[i] or rsi_val > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (trend reversal) or overextended RSI
            if price > r1_aligned[i] or rsi_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals