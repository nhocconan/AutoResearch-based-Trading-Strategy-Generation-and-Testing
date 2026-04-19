#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily RSI(14) with weekly trend filter (EMA21) and volume confirmation.
# In bull markets: RSI > 55 + price above weekly EMA21 + volume > 1.5x average = long.
# In bear markets: RSI < 45 + price below weekly EMA21 + volume > 1.5x average = short.
# Weekly EMA21 filters trend direction to avoid counter-trend trades.
# Volume confirmation ensures breakouts have conviction.
# Designed for 1d timeframe to capture multi-day moves with low frequency (target: 15-25 trades/year).
# Uses discrete position sizes (0.0, ±0.25) to minimize fee churn.
name = "1d_RSI_WeeklyEMA_Volume"
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
    
    # Weekly EMA21 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Weekly EMA21
    weekly_close = df_1w['close'].values
    ema21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_aligned = align_htf_to_ltf(prices, df_1w, ema21)
    
    # Daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: volume > 1.5 * 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure enough data for all indicators (21 for weekly EMA + 14 for RSI)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI > 55, price above weekly EMA21, volume confirmation
            if (rsi[i] > 55 and 
                close[i] > ema21_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 45, price below weekly EMA21, volume confirmation
            elif (rsi[i] < 45 and 
                  close[i] < ema21_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if RSI < 40 or price crosses below weekly EMA21
            if (rsi[i] < 40) or (close[i] < ema21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if RSI > 60 or price crosses above weekly EMA21
            if (rsi[i] > 60) or (close[i] > ema21_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals