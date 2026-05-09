#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w trend filter and 1d price action for entries.
# Uses 1w EMA20 for trend filter and 1d RSI(14) for momentum confirmation.
# Designed for low trade frequency (7-25/year) to avoid fee drag in 1d timeframe.
# Works in both bull/bear markets by requiring alignment with 1w trend and momentum confirmation.
name = "1d_RSI14_1wEMA20_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 1d RSI(14) for momentum
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
    
    start_idx = 20  # Wait for EMA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_20_1d[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Momentum filters
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        
        if position == 0:
            # Long: price above 1w EMA20 and RSI oversold
            if close[i] > ema_20_1d[i] and rsi_oversold:
                signals[i] = 0.25
                position = 1
            # Short: price below 1w EMA20 and RSI overbought
            elif close[i] < ema_20_1d[i] and rsi_overbought:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 1w EMA20 or RSI overbought
            if close[i] < ema_20_1d[i] or rsi_values[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 1w EMA20 or RSI oversold
            if close[i] > ema_20_1d[i] or rsi_values[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals