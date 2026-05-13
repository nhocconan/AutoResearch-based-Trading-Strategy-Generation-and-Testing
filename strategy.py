# The strategy is a 1-hour trend-following system that uses 4h and 1d higher timeframes for trend direction and regime filtering. It enters long when the 4h EMA50 is above the 1d EMA200 (bullish regime) and price pulls back to the 4h EMA20 with momentum confirmation (RSI > 50). It enters short when the 4h EMA50 is below the 1d EMA200 (bearish regime) and price rallies to the 4h EMA20 with momentum confirmation (RSI < 50). Exits occur when the price crosses the 4h EMA50 or momentum diverges (RSI crosses 50 in the opposite direction). The strategy uses a fixed position size of 0.20 to limit risk and includes an 8 AM to 8 PM UTC session filter to avoid low-liquidity periods. Designed to capture medium-term trends while avoiding whipsaws in ranging markets, it should perform in both bull and bear regimes by aligning with the higher timeframe trend.

#!/usr/bin/env python3
name = "1h_EMA_Trend_Follow_With_Momentum_Filter"
timeframe = "1h"
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

    # Get 4h and 1h data for trend and execution
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')

    # Calculate 4h EMA50 and EMA20 for trend and entry
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)

    # Calculate 1d EMA200 for regime filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    # Calculate 1h RSI for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values

    # Session filter: 8 AM to 8 PM UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Only trade during session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bullish regime (4h EMA50 > 1d EMA200) + price at 4h EMA20 support + bullish momentum (RSI > 50)
            if ema50_4h_aligned[i] > ema200_1d_aligned[i] and \
               close[i] <= ema20_4h_aligned[i] * 1.001 and \
               rsi[i] > 50:
                signals[i] = 0.20
                position = 1
            # SHORT: Bearish regime (4h EMA50 < 1d EMA200) + price at 4h EMA20 resistance + bearish momentum (RSI < 50)
            elif ema50_4h_aligned[i] < ema200_1d_aligned[i] and \
                 close[i] >= ema20_4h_aligned[i] * 0.999 and \
                 rsi[i] < 50:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish regime shift or momentum failure
            if ema50_4h_aligned[i] < ema200_1d_aligned[i] or rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Bullish regime shift or momentum failure
            if ema50_4h_aligned[i] > ema200_1d_aligned[i] or rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals