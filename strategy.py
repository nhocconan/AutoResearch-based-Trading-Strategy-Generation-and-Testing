#!/usr/bin/env python3
name = "1d_1w_RSI_Overbought_Oversold"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Weekly RSI(14) - overbought >70, oversold <30
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    
    # Daily trend filter: EMA(50) on daily close
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # Wait for EMA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(ema_50[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly RSI oversold (<30) and price above daily EMA50
            if rsi_1w_aligned[i] < 30 and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly RSI overbought (>70) and price below daily EMA50
            elif rsi_1w_aligned[i] > 70 and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or price below EMA50
            if rsi_1w_aligned[i] > 40 or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or price above EMA50
            if rsi_1w_aligned[i] < 60 or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Daily RSI overbought/oversold with weekly trend filter
# - Weekly RSI < 30 (oversold) + price above daily EMA50 = long opportunity
# - Weekly RSI > 70 (overbought) + price below daily EMA50 = short opportunity
# - Weekly RSI provides higher timeframe extreme readings, reducing false signals
# - Daily EMA50 acts as trend filter - only trade in direction of weekly extreme
# - Exit when RSI returns to neutral territory (40-60) or price crosses EMA50
# - Position size 0.25 targets ~15-25 trades/year, minimizing fee drag
# - Works in both bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)