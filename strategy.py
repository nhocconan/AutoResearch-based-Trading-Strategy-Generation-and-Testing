#!/usr/bin/env python3
# 1d_RSI_MeanReversion_BollingerBand_Exit
# Hypothesis: In a bear/ranging market (2025+), RSI extremes on daily timeframe provide mean-reversion opportunities. 
# Enter long when RSI < 30 and price below lower Bollinger Band, short when RSI > 70 and price above upper Bollinger Band.
# Exit when price crosses back to the middle Bollinger Band (20-day SMA). 
# Uses weekly trend filter: only take longs when price > weekly EMA50, shorts when price < weekly EMA50 to avoid counter-trend trades.
# Bollinger Band squeeze (low volatility) acts as additional filter to avoid choppy markets.
# Target: 10-25 trades/year to minimize fee drag, works in both bull/bear via trend filter and volatility regime.

name = "1d_RSI_MeanReversion_BollingerBand_Exit"
timeframe = "1d"
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
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate RSI(14) on daily
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    # Bollinger Bands (20, 2)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    lower_bb = sma20 - 2 * std20
    upper_bb = sma20 + 2 * std20

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Bollinger Band width for volatility regime (avoid chop)
    bb_width = (upper_bb - lower_bb) / sma20
    bb_width_avg_20 = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(rsi[i]) or np.isnan(sma20[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(upper_bb[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(bb_width_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Volatility filter: avoid extremely low volatility (chop)
        if bb_width[i] < bb_width_avg_20[i] * 0.5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: RSI oversold + price below lower BB + above weekly EMA50 (uptrend filter)
            if (rsi[i] < 30 and 
                close[i] < lower_bb[i] and
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought + price above upper BB + below weekly EMA50 (downtrend filter)
            elif (rsi[i] > 70 and 
                  close[i] > upper_bb[i] and
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses above middle Bollinger Band (SMA20)
            if close[i] > sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses below middle Bollinger Band (SMA20)
            if close[i] < sma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals