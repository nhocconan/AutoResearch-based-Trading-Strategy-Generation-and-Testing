#!/usr/bin/env python3
# 6h_Relative_Strength_Index_Divergence_Trend_Filter
# Hypothesis: Enter long when RSI(14) shows bullish divergence (price makes lower low, RSI makes higher low) 
# with price above 200-period EMA and bullish weekly trend. Enter short on bearish divergence with price 
# below 200 EMA and bearish weekly trend. Divergence signals potential reversals with reduced false signals 
# in trending markets. Weekly trend filter ensures alignment with higher timeframe momentum.
# Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend). Low frequency due to strict 
# divergence requirement and trend filter.

name = "6h_Relative_Strength_Index_Divergence_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly trend: EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 200-period EMA for trend filter
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema200[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Check for bullish divergence: price lower low, RSI higher low
            bullish_div = False
            if i >= 20:  # Need sufficient lookback
                # Look for recent swing low in price
                for lookback in range(5, min(20, i)):
                    if low[i] < low[i-lookback] and rsi[i] > rsi[i-lookback]:
                        # Confirm with higher low in RSI
                        bullish_div = True
                        break
            
            # Check for bearish divergence: price higher high, RSI lower high
            bearish_div = False
            if i >= 20:  # Need sufficient lookback
                # Look for recent swing high in price
                for lookback in range(5, min(20, i)):
                    if high[i] > high[i-lookback] and rsi[i] < rsi[i-lookback]:
                        # Confirm with lower high in RSI
                        bearish_div = True
                        break

            # LONG: Bullish divergence + price above EMA200 + weekly uptrend
            if bullish_div and close[i] > ema200[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence + price below EMA200 + weekly downtrend
            elif bearish_div and close[i] < ema200[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below EMA200 or bearish divergence
            exit_condition = False
            if i >= 20:
                for lookback in range(5, min(20, i)):
                    if high[i] > high[i-lookback] and rsi[i] < rsi[i-lookback]:
                        exit_condition = True
                        break
            if close[i] < ema200[i] or exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above EMA200 or bullish divergence
            exit_condition = False
            if i >= 20:
                for lookback in range(5, min(20, i)):
                    if low[i] < low[i-lookback] and rsi[i] > rsi[i-lookback]:
                        exit_condition = True
                        break
            if close[i] > ema200[i] or exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals