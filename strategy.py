#!/usr/bin/env python3
"""
1D_Weekly_Pivot_Reversion_Mean
Hypothesis: In ranging markets, price tends to revert from weekly pivot resistance/support levels.
Long when price closes below weekly S1 with bullish engulfing candle and RSI < 40.
Short when price closes above weekly R1 with bearish engulfing candle and RSI > 60.
Uses 1d timeframe with 1h pivot confirmation to avoid false breaks. Designed for mean reversion in both bull and bear markets.
"""
name = "1D_Weekly_Pivot_Reversion_Mean"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points (R1, S1, PP)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    pp = (high_weekly + low_weekly + close_weekly) / 3
    range_weekly = high_weekly - low_weekly
    r1 = pp + (range_weekly * 1.1 / 12)
    s1 = pp - (range_weekly * 1.1 / 12)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    
    # RSI(14) for momentum filter
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Engulfing candle detection
    bullish_engulf = (close > open_price) & (open_price > close.shift(1)) & (close > close.shift(1)) & (open_price < open_price.shift(1))
    bearish_engulf = (close < open_price) & (open_price < close.shift(1)) & (close < close.shift(1)) & (open_price > open_price.shift(1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # RSI warmup
    
    for i in range(start_idx, n):
        # Skip if pivot data not ready
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price below S1, bullish engulf, oversold RSI
            if (close[i] < s1_aligned[i] and 
                bullish_engulf[i] and 
                rsi_values[i] < 40):
                signals[i] = 0.25
                position = 1
            # Short setup: price above R1, bearish engulf, overbought RSI
            elif (close[i] > r1_aligned[i] and 
                  bearish_engulf[i] and 
                  rsi_values[i] > 60):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit conditions: RSI mean reversion or opposite engulfing
            if position == 1 and (rsi_values[i] > 60 or bearish_engulf[i]):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (rsi_values[i] < 40 or bullish_engulf[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals