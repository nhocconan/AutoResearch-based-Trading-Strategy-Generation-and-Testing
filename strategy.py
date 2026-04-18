#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h RSI(2) pullback with 1d trend filter. Works in bull (pullbacks in uptrend) and bear (short rallies in downtrend).
# Uses extremely short RSI for mean reversion entries, filtered by daily trend to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
name = "6h_RSI2_Pullback_1dTrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) in uptrend (price > daily EMA50)
            if rsi_val < 10 and close_val > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI(2) > 90 (overbought) in downtrend (price < daily EMA50)
            elif rsi_val > 90 and close_val < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI(2) > 50 (mean reversion complete) or trend breaks
            if rsi_val > 50 or close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI(2) < 50 (mean reversion complete) or trend breaks
            if rsi_val < 50 or close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals