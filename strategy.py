#!/usr/bin/env python3
"""
12h_1D_RSI_CCI_MeanReversion_v1
Concept: Use 1-day RSI and CCI for mean reversion on 12h timeframe.
- Long: RSI(14) < 30 AND CCI(20) < -100
- Short: RSI(14) > 70 AND CCI(20) > 100
- Exit: RSI crosses back to 50 (mean reversion)
- Position sizing: 0.25
- Works in bull/bear: Mean reversion works in ranging and trending markets
- Low trade frequency: Target 10-30 trades/year on 12h
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1D_RSI_CCI_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for RSI and CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h: Price data ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Daily: RSI(14) ===
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_values[:14] = np.nan  # First 14 values are invalid
    
    # === Daily: CCI(20) ===
    tp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3  # Typical price
    sma_tp = tp.rolling(window=20, min_periods=20).mean()
    mad = tp.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad)
    cci_values = cci.values
    cci_values[:20] = np.nan  # First 20 values are invalid
    
    # Align daily indicators to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        rsi_val = rsi_aligned[i]
        cci_val = cci_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(rsi_val) or np.isnan(cci_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold AND CCI deeply oversold
            if rsi_val < 30 and cci_val < -100:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought AND CCI deeply overbought
            elif rsi_val > 70 and cci_val > 100:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (50)
            if rsi_val >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (50)
            if rsi_val <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals