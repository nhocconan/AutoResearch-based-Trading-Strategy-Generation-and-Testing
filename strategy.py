#!/usr/bin/env python3
# 6H_RSI_DIVERGENCE_1D_TREND_FILTER
# Hypothesis: RSI divergence on 1d timeframe signals reversals in the primary trend.
# In 1d uptrend (price > EMA50), look for bearish RSI divergence (price higher high, RSI lower high) to go short.
# In 1d downtrend (price < EMA50), look for bullish RSI divergence (price lower low, RSI higher low) to go long.
# Works in both bull and bear markets: trend filter ensures trades align with higher timeframe momentum,
# while RSI divergence captures exhaustion points for high-probability reversals.
# Target: 15-25 trades/year on 6h timeframe.

name = "6H_RSI_DIVERGENCE_1D_TREND_FILTER"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Daily data for RSI and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # RSI(14) calculation
    rsi = calculate_rsi(df_1d['close'].values, 14)
    
    # EMA50 for trend filter
    ema50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Get values for divergence check (need at least 3 bars back)
            if i >= 3:
                rsi_now = rsi_aligned[i]
                rsi_prev = rsi_aligned[i-1]
                rsi_prev2 = rsi_aligned[i-2]
                price_now = close[i]
                price_prev = close[i-1]
                price_prev2 = close[i-2]
                
                # Bullish divergence: price makes lower low, RSI makes higher low
                bullish_div = (price_now < price_prev and price_prev < price_prev2 and
                              rsi_now > rsi_prev and rsi_prev > rsi_prev2)
                
                # Bearish divergence: price makes higher high, RSI makes lower high
                bearish_div = (price_now > price_prev and price_prev > price_prev2 and
                              rsi_now < rsi_prev and rsi_prev < rsi_prev2)
                
                # LONG: 1d uptrend + bullish RSI divergence
                if (price_now > ema50_aligned[i] and bullish_div):
                    signals[i] = 0.25
                    position = 1
                # SHORT: 1d downtrend + bearish RSI divergence
                elif (price_now < ema50_aligned[i] and bearish_div):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or RSI overbought
            if (price_now := close[i]) <= ema50_aligned[i] or rsi_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or RSI oversold
            if (price_now := close[i]) >= ema50_aligned[i] or rsi_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals