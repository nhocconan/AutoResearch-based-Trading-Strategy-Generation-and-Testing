#!/usr/bin/env python3
"""
12h_4h_Trend_1d_Momentum_v1
Hypothesis: Use 4h EMA50/200 for trend direction, combined with 12h RSI(14) for momentum timing.
Long when 12h price > 4h EMA50 AND 4h EMA50 > 4h EMA200 (uptrend) AND 12h RSI > 55.
Short when 12h price < 4h EMA50 AND 4h EMA50 < 4h EMA200 (downtrend) AND 12h RSI < 45.
Exit when trend condition fails or RSI reverts to neutral (45-55).
This captures trend-following entries with momentum confirmation, avoiding whipsaws in sideways markets.
Target: 20-50 trades over 4 years on 12h timeframe.
"""

name = "12h_4h_Trend_1d_Momentum_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4H Data for EMA Trend ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:  # Need enough data for EMA200
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # EMA50 and EMA200 on 4h
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h EMAs to 12h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # === 12H Data for RSI Momentum ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # fill NaN with 50 (neutral)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA200
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema200_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend condition: EMA50 > EMA200 = uptrend, EMA50 < EMA200 = downtrend
        ema50 = ema50_4h_aligned[i]
        ema200 = ema200_4h_aligned[i]
        uptrend = ema50 > ema200
        downtrend = ema50 < ema200
        
        # Price vs EMA50 condition
        price_above_ema50 = close[i] > ema50
        price_below_ema50 = close[i] < ema50
        
        # RSI momentum condition
        rsi_val = rsi_values[i]
        rsi_bullish = rsi_val > 55
        rsi_bearish = rsi_val < 45
        
        if position == 0:
            # Long: uptrend AND price above EMA50 AND RSI bullish
            if uptrend and price_above_ema50 and rsi_bullish:
                signals[i] = 0.25
                position = 1
            # Short: downtrend AND price below EMA50 AND RSI bearish
            elif downtrend and price_below_ema50 and rsi_bearish:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend fails OR RSI reverts to neutral
            if not (uptrend and price_above_ema50) or not rsi_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: trend fails OR RSI reverts to neutral
            if not (downtrend and price_below_ema50) or not rsi_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals