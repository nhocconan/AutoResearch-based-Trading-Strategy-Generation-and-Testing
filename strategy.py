#!/usr/bin/env python3
"""
1d_1w_Weekly_Pullback_Reversal
Hypothesis: Trade pullbacks to weekly support/resistance on 1d timeframe with RSI confirmation.
In bull markets, price pulls back to weekly low (support) before continuing up.
In bear markets, price pulls back to weekly high (resistance) before continuing down.
RSI(2) identifies overextended pullbacks for high-probability reversal entries.
Weekly levels are strong institutional reference points that hold across regimes.
Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for support/resistance levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Align weekly high/low to 1d
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, high_weekly)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, low_weekly)
    
    # RSI(2) for short-term overextension
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Long: pullback to weekly low with RSI < 30 (oversold)
        long_condition = (low[i] <= weekly_low_aligned[i] * 1.002) and (rsi_values[i] < 30)
        
        # Short: pullback to weekly high with RSI > 70 (overbought)
        short_condition = (high[i] >= weekly_high_aligned[i] * 0.998) and (rsi_values[i] > 70)
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_Weekly_Pullback_Reversal"
timeframe = "1d"
leverage = 1.0