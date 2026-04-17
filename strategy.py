# 6H_1D_RSI_MEANREV_OVERSOLD

#!/usr/bin/env python3
"""
Hypothesis: On 6-hour timeframe, RSI(14) extreme readings with daily trend filter
capture mean-reversion opportunities in both bull and bear markets.
- Long when RSI < 30 and price > daily EMA50 (oversold in uptrend)
- Short when RSI > 70 and price < daily EMA50 (overbought in downtrend)
- Exit when RSI returns to neutral zone (40-60)
Uses daily EMA50 for trend filter to avoid counter-trend trades.
Designed for ~25-40 trades/year with discrete sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for EMA50 and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily RSI(14)
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate daily EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 6h timeframe
    rsi_6h = align_htf_to_ltf(prices, df_1d, rsi_values)
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for daily indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi_6h[i]) or np.isnan(ema_50_6h[i]):
            signals[i] = 0.0
            continue
            
        price = close[i]
        rsi_val = rsi_6h[i]
        ema_val = ema_50_6h[i]
        
        if position == 0:
            # Long: RSI oversold (<30) and price above daily EMA50 (uptrend)
            if rsi_val < 30 and price > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought (>70) and price below daily EMA50 (downtrend)
            elif rsi_val > 70 and price < ema_val:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: RSI returns to neutral (>=40) or turns overbought
            if rsi_val >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: RSI returns to neutral (<=60) or turns oversold
            if rsi_val <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6H_1D_RSI_MEANREV_OVERSOLD"
timeframe = "6h"
leverage = 1.0