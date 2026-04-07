#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h RSI Mean Reversion with Daily Trend Filter
# Hypothesis: RSI extremes on 12h timeframe combined with daily trend direction
# provides high-probability mean reversion entries in both bull and bear markets.
# In uptrend, buy RSI<30; in downtrend, sell RSI>70. Avoids counter-trend trades.
# Target: 20-30 trades/year (80-120 total over 4 years).

name = "12h_rsi_mean_reversion_1d_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 or trend turns bearish
            if rsi_values[i] > 50 or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 or trend turns bullish
            if rsi_values[i] < 50 or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade in direction of daily trend
            if close[i] > ema_50_1d_aligned[i]:  # Uptrend
                if rsi_values[i] < 30:  # Oversold - buy the dip
                    position = 1
                    signals[i] = 0.25
            elif close[i] < ema_50_1d_aligned[i]:  # Downtrend
                if rsi_values[i] > 70:  # Overbought - sell the rally
                    position = -1
                    signals[i] = -0.25
    
    return signals