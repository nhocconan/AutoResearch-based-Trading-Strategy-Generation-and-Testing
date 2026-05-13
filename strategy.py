#!/usr/bin/env python3
# 12h_KAMA_Trend_With_RSI_Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) captures trend with low lag, combined with RSI for momentum confirmation on 12h timeframe.
# Uses 1-day timeframe for trend filter (EMA34) and volume confirmation to reduce false signals.
# Designed for 12-30 trades/year to minimize fee drag while capturing major moves.
# Works in bull/bear: long when KAMA turns up with RSI>50 and price above daily EMA; short when KAMA turns down with RSI<50 and price below daily EMA.

name = "12h_KAMA_Trend_With_RSI_Filter"
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

    # Get daily data for trend filter and volume context
    df_1d = get_htf_data(prices, '1d')

    # KAMA parameters
    er_length = 10
    fast_ema = 2
    slow_ema = 30

    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle first er_length elements
    change = np.concatenate([np.full(er_length, np.nan), change])
    volatility = np.concatenate([np.full(er_length, np.nan), volatility[er_length:]])
    # Calculate rolling sum of volatility
    volatility_sum = np.convolve(volatility, np.ones(er_length), 'same')
    volatility_sum[:er_length-1] = np.nan
    volatility_sum[-er_length+1:] = np.nan
    er = np.where(volatility_sum != 0, change / volatility_sum, 0)
    er = np.nan_to_num(er, nan=0.0)

    # Smoothing constant
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_length-1] = close[er_length-1]  # seed
    for i in range(er_length, n):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = close[i]

    # 1-day EMA34 trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # RSI (14) for momentum
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.convolve(gain, np.ones(14)/14, 'same')
    avg_loss = np.convolve(loss, np.ones(14)/14, 'same')
    avg_gain[:13] = np.nan
    avg_loss[:13] = np.nan
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50.0)

    # Volume confirmation: current volume > 1.5 x 20-period average (adapted for 12h)
    vol_ma = np.convolve(volume, np.ones(20)/20, 'same')
    vol_ma[:19] = np.nan
    vol_ma[-19:] = np.nan
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: KAMA rising, RSI > 50, price above daily EMA34, with volume confirmation
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI < 50, price below daily EMA34, with volume confirmation
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or price closes below daily EMA34
            if kama[i] < kama[i-1] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or price closes above daily EMA34
            if kama[i] > kama[i-1] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals