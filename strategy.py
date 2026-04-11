#!/usr/bin/env python3
"""
1d_1w_rsi_reversal_volume
Strategy: 1-day RSI reversal with 1-week volume confirmation
Timeframe: 1d
Leverage: 1.0
Hypothesis: Uses weekly volume expansion (current week volume > 1.5x 4-week average) to confirm daily RSI reversals (RSI < 30 for long, RSI > 70 for short). Designed for low trade frequency (<15/year) to minimize fee decay while capturing mean-reversion moves in both bull and bear markets. Works in trending markets via pullbacks and in ranging markets via extreme reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_reversal_volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Daily RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly volume expansion filter
    vol_1w = df_1w['volume'].values
    vol_ma_4 = pd.Series(vol_1w).rolling(window=4, min_periods=4).mean().values
    vol_ratio_1w = vol_1w / vol_ma_4
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if weekly volume data is invalid
        if np.isnan(vol_ratio_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        rsi_val = rsi[i]
        vol_expanded = vol_ratio_1w_aligned[i] > 1.5
        
        # Long: RSI oversold with volume confirmation
        long_signal = rsi_val < 30 and vol_expanded
        # Short: RSI overbought with volume confirmation
        short_signal = rsi_val > 70 and vol_expanded
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and rsi_val > 40
        exit_short = position == -1 and rsi_val < 60
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals