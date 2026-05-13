#!/usr/bin/env python3
# 12h_RSI_Extreme_With_Trend_Filter
# Hypothesis: RSI extremes on 12h timeframe combined with 1-day trend filter (EMA50) 
# captures mean-reversion in ranging markets and trend continuation in strong trends.
# Long when RSI < 30 and price above daily EMA50 (bullish bias).
# Short when RSI > 70 and price below daily EMA50 (bearish bias).
# Exit when RSI returns to neutral range (40-60).
# Works in bull markets by buying dips in uptrends, and in bear markets by selling rallies in downtrends.

name = "12h_RSI_Extreme_With_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA50 calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA50 on 1d close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate RSI on 12h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(14, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold (<30) and price above 1d EMA50 (bullish bias)
            if rsi[i] < 30 and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) and price below 1d EMA50 (bearish bias)
            elif rsi[i] > 70 and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>=40)
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<=60)
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals