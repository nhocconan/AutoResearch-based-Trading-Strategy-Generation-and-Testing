# 1D_KAMA_RSI_ChopFilter_v1
# Hypothesis: Uses KAMA (adaptive trend) on daily timeframe for direction, RSI for entry timing, and Choppiness Index for regime filtering. Enters long when KAMA is rising (uptrend), RSI crosses above 30 from oversold, and market is not choppy (CHOP > 61.8). Enters short when KAMA is falling (downtrend), RSI crosses below 70 from overbought, and market is not choppy. Exits when trend reverses or RSI reaches opposite extreme. Designed for low trade frequency (<25/year) with high win rate in both bull and bear markets by avoiding choppy regimes and using adaptive trend following.

name = "1D_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
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
    volume = prices['volume'].values

    # Get weekly data for higher timeframe trend filter (optional but helpful)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)

    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Get daily data for KAMA, RSI, and Choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for indicators
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate KAMA (adaptive moving average) on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period sum of absolute changes
    # Handle first 10 values where diff doesn't work
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start at index 9
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)

    # Calculate RSI (14) on daily close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.concatenate([np.full(14, np.nan), [np.mean(gain[:14])]])
    avg_loss = np.concatenate([np.full(14, np.nan), [np.mean(loss[:14])]])
    for i in range(15, len(close_1d)):
        avg_gain = np.append(avg_gain, (avg_gain[-1] * 13 + gain[i-1]) / 14)
        avg_loss = np.append(avg_loss, (avg_loss[-1] * 13 + loss[i-1]) / 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)

    # Calculate Choppiness Index (14) on daily data
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    # Sum of TR over 14 periods
    atr_sum = np.convolve(tr, np.ones(14), 'valid')
    atr_sum = np.concatenate([np.full(13, np.nan), atr_sum])
    # Highest high and lowest low over 14 periods
    hh = np.maximum.accumulate(high_1d)
    ll = np.minimum.accumulate(low_1d)
    hh14 = np.concatenate([np.full(13, np.nan), hh[13:]])
    ll14 = np.concatenate([np.full(13, np.nan), ll[13:]])
    # Chop calculation
    chop = 100 * np.log10(atr_sum / (hh14 - ll14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Regime filter: only trade when not choppy (CHOP > 61.8 = trending)
        not_choppy = chop_aligned[i] > 61.8

        if position == 0:
            # LONG: KAMA rising (uptrend), RSI crossing above 30 from oversold, not choppy, and above weekly EMA
            kama_rising = kama_aligned[i] > kama_aligned[i-1]
            rsi_cross_up = rsi_aligned[i] > 30 and rsi_aligned[i-1] <= 30
            above_weekly_ema = close[i] > ema20_1w_aligned[i]
            if kama_rising and rsi_cross_up and not_choppy and above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (downtrend), RSI crossing below 70 from overbought, not choppy, and below weekly EMA
            elif (kama_aligned[i] < kama_aligned[i-1] and 
                  rsi_aligned[i] < 70 and rsi_aligned[i-1] >= 70 and 
                  not_choppy and close[i] < ema20_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling (trend change) or RSI overbought (70) or choppy market
            kama_falling = kama_aligned[i] < kama_aligned[i-1]
            rsi_overbought = rsi_aligned[i] >= 70
            if kama_falling or rsi_overbought or not not_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising (trend change) or RSI oversold (30) or choppy market
            kama_rising = kama_aligned[i] > kama_aligned[i-1]
            rsi_oversold = rsi_aligned[i] <= 30
            if kama_rising or rsi_oversold or not not_choppy:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals