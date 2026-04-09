#!/usr/bin/env python3
# mtf_1h_ema_rsi_pullback_4h1d_v1
# Hypothesis: 1h strategy using 4h EMA(21) for trend direction and 1d RSI(14) for regime filter.
# Enters long on 1h EMA(8)/EMA(21) bullish crossover when price > 4h EMA(21) and 1d RSI < 70.
# Enters short on 1h EMA(8)/EMA(21) bearish crossover when price < 4h EMA(21) and 1d RSI > 30.
# Uses discrete sizing (0.20) to limit fee churn. Target: 15-37 trades/year (60-150 total over 4 years).
# Works in bull/bear: 4h EMA filters trend, 1d RSI avoids overextended entries, 1h EMA cross provides timely entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_ema_rsi_pullback_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h EMA(8) and EMA(21) for entry timing
    close_s = pd.Series(close)
    ema_8 = close_s.ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Multi-timeframe: 4h EMA(21) trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    close_4h_s = pd.Series(close_4h)
    ema_21_4h = close_4h_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Multi-timeframe: 1d RSI(14) regime filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    delta = close_1d_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = (100 - (100 / (1 + rs))).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_8[i]) or np.isnan(ema_21[i]) or np.isnan(ema_21_4h_aligned[i]) or
            np.isnan(rsi_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # 1h EMA cross signals
        ema_bullish_cross = ema_8[i] > ema_21[i] and ema_8[i-1] <= ema_21[i-1]
        ema_bearish_cross = ema_8[i] < ema_21[i] and ema_8[i-1] >= ema_21[i-1]
        
        if position == 1:  # Long position
            # Exit: EMA bearish cross or price < 4h EMA(21)
            if ema_bearish_cross or close[i] < ema_21_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: EMA bullish cross or price > 4h EMA(21)
            if ema_bullish_cross or close[i] > ema_21_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Check for EMA cross with trend and regime filters
            bullish_entry = ema_bullish_cross and close[i] > ema_21_4h_aligned[i] and rsi_14_1d_aligned[i] < 70
            bearish_entry = ema_bearish_cross and close[i] < ema_21_4h_aligned[i] and rsi_14_1d_aligned[i] > 30
            
            if bullish_entry:
                position = 1
                signals[i] = 0.20
            elif bearish_entry:
                position = -1
                signals[i] = -0.20
    
    return signals