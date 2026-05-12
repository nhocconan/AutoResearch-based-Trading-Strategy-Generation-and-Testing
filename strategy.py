#!/usr/bin/env python3
# 6h_1w_1d_RSI_Divergence_Pullback
# Hypothesis: On 6h timeframe, enter long when RSI shows bullish divergence (price makes lower low, RSI makes higher low) during pullback to 21-period EMA in a weekly uptrend; enter short on bearish divergence during pullback to EMA in weekly downtrend. Uses daily EMA200 as additional trend filter. Designed for low frequency (15-35 trades/year) to avoid fee drag. Works in bull/bear by following weekly trend and using mean-reversion entries during pullbacks.

name = "6h_1w_1d_RSI_Divergence_Pullback"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend direction
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Get daily data for EMA200 filter and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values
    # Daily EMA200 for additional trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    # Calculate RSI on 6h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # Calculate 6h EMA21 for pullback entries
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(ema21[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Determine weekly trend
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]

        # Additional daily trend filter
        daily_filter_long = close[i] > ema200_1d_aligned[i]
        daily_filter_short = close[i] < ema200_1d_aligned[i]

        if position == 0:
            # Check for bullish RSI divergence: price lower low, RSI higher low
            bullish_div = False
            if i >= 20:  # Need lookback for divergence
                # Find recent swing low in price
                lookback = 20
                price_lows = []
                rsi_lows = []
                for j in range(i - lookback, i):
                    if j >= 1 and j < n-1:
                        if low[j] <= low[j-1] and low[j] <= low[j+1]:
                            price_lows.append((j, low[j]))
                            rsi_lows.append((j, rsi[j]))
                # Check for at least two lows where price makes lower low but RSI makes higher low
                if len(price_lows) >= 2:
                    price_lows.sort(key=lambda x: x[0])  # Sort by time
                    rsi_lows.sort(key=lambda x: x[0])
                    # Compare last two lows
                    if (price_lows[-1][1] < price_lows[-2][1] and 
                        rsi_lows[-1][1] > rsi_lows[-2][1]):
                        bullish_div = True

            # Check for bearish RSI divergence: price higher high, RSI lower high
            bearish_div = False
            if i >= 20:
                lookback = 20
                price_highs = []
                rsi_highs = []
                for j in range(i - lookback, i):
                    if j >= 1 and j < n-1:
                        if high[j] >= high[j-1] and high[j] >= high[j+1]:
                            price_highs.append((j, high[j]))
                            rsi_highs.append((j, rsi[j]))
                if len(price_highs) >= 2:
                    price_highs.sort(key=lambda x: x[0])
                    rsi_highs.sort(key=lambda x: x[0])
                    if (price_highs[-1][1] > price_highs[-2][1] and 
                        rsi_highs[-1][1] < rsi_highs[-2][1]):
                        bearish_div = True

            # LONG: Bullish divergence during pullback to EMA21 in weekly uptrend + daily filter
            if (bullish_div and weekly_uptrend and daily_filter_long and 
                close[i] <= ema21[i] * 1.005):  # Allow small buffer above EMA
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence during pullback to EMA21 in weekly downtrend + daily filter
            elif (bearish_div and weekly_downtrend and daily_filter_short and 
                  close[i] >= ema21[i] * 0.995):  # Allow small buffer below EMA
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes above EMA21 (momentum) or RSI > 70 (overbought)
            if close[i] > ema21[i] * 1.01 or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes below EMA21 (momentum) or RSI < 30 (oversold)
            if close[i] < ema21[i] * 0.99 or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals