#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d trend and 12h momentum for entries.
# Uses 1d EMA50 for trend filter and 12h RSI(14) for momentum confirmation.
# Designed for low trade frequency (12-37/year) to avoid fee drag in 12h timeframe.
# Works in both bull/bear markets by requiring alignment with 1d trend and momentum confirmation.
name = "12h_RSI14_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h RSI(14) for momentum
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_12h[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Momentum filters
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        
        if position == 0:
            # Long: price above 1d EMA50 and RSI oversold
            if close[i] > ema_50_12h[i] and rsi_oversold:
                signals[i] = 0.25
                position = 1
            # Short: price below 1d EMA50 and RSI overbought
            elif close[i] < ema_50_12h[i] and rsi_overbought:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 1d EMA50 or RSI overbought
            if close[i] < ema_50_12h[i] or rsi_values[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 1d EMA50 or RSI oversold
            if close[i] > ema_50_12h[i] or rsi_values[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals