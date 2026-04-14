#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w EMA trend filter and 1d RSI mean reversion
# In both bull and bear markets, RSI extremes often reverse when aligned with the weekly trend
# Long when RSI < 30 and price above weekly EMA50 (uptrend)
# Short when RSI > 70 and price below weekly EMA50 (downtrend)
# Uses weekly EMA for trend filter to avoid counter-trend trades
# Designed for low frequency (target 10-30 trades/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load 1w data ONCE for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA(50)
    ema_length = 50
    ema_source = df_1w['close'].values
    ema_values = pd.Series(ema_source).ewm(span=ema_length, adjust=False, min_periods=ema_length).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_values)
    
    # Calculate 1d RSI(14)
    rsi_length = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_length, adjust=False, min_periods=rsi_length).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_length, adjust=False, min_periods=rsi_length).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, rsi_length, ema_length)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_val = ema_aligned[i]
        
        if position == 0:
            # Enter long: oversold RSI + price above weekly EMA (uptrend)
            if rsi_val < 30 and price > ema_val:
                position = 1
                signals[i] = position_size
            # Enter short: overbought RSI + price below weekly EMA (downtrend)
            elif rsi_val > 70 and price < ema_val:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (50) or price crosses below weekly EMA
            if rsi_val >= 50 or price < ema_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (50) or price crosses above weekly EMA
            if rsi_val <= 50 or price > ema_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wEMA50_RSI14_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0