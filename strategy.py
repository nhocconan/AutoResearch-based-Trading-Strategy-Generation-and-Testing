# 1d_RSI_Trend_Filtered
# Hypothesis: On daily timeframe, RSI(14) with trend filter (EMA200) and volume confirmation reduces false signals in both bull and bear markets.
# Trend filter ensures we only trade in direction of long-term trend, improving win rate.
# Volume confirmation ensures breakouts have conviction.
# Target: 15-25 trades/year to minimize fee drag.

name = "1d_RSI_Trend_Filtered"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)

    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)

    # Calculate daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # Calculate daily volume SMA20 for confirmation
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI > 55 (bullish momentum) + price above EMA200 (uptrend) + volume confirmation
            if rsi[i] > 55 and close[i] > ema200_1w_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI < 45 (bearish momentum) + price below EMA200 (downtrend) + volume confirmation
            elif rsi[i] < 45 and close[i] < ema200_1w_aligned[i] and volume[i] > volume_sma20[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 40 (loss of momentum) or price below EMA200 (trend change)
            if rsi[i] < 40 or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI > 60 (loss of bearish momentum) or price above EMA200 (trend change)
            if rsi[i] > 60 or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals