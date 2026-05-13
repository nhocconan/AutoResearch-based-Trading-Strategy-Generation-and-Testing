#!/usr/bin/env python3
"""
1d_RSI_Reversal_With_Weekly_Trend_Filter
Hypothesis: RSI mean-reversion (RSI<30 for long, RSI>70 for short) combined with weekly trend filter (price above/below weekly EMA50) works in both bull and bear markets. Uses 25% position size to limit risk and target ~15-25 trades/year on daily timeframe to minimize fee drag.
"""

name = "1d_RSI_Reversal_With_Weekly_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate RSI(14) on close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # fill neutral before warmup
    
    # Weekly trend filter: EMA(50) on close
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after RSI warmup
        if position == 0:
            # LONG: RSI crosses above 30 from below, price above weekly EMA50 (uptrend filter)
            if (rsi_values[i] > 30 and rsi_values[i-1] <= 30 and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI crosses below 70 from above, price below weekly EMA50 (downtrend filter)
            elif (rsi_values[i] < 70 and rsi_values[i-1] >= 70 and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses above 70 (overbought) or price crosses below weekly EMA50
            if (rsi_values[i] >= 70 and rsi_values[i-1] < 70) or \
               (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses below 30 (oversold) or price crosses above weekly EMA50
            if (rsi_values[i] <= 30 and rsi_values[i-1] > 30) or \
               (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals