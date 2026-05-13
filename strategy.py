#!/usr/bin/env python3
# 12h_KAMA_Trend_RSI_Chop_Filter
# Hypothesis: KAMA adapts to market efficiency, providing a robust trend signal. 
# Long when price > KAMA, RSI > 50, and Choppiness Index < 38.2 (trending market). 
# Short when price < KAMA, RSI < 50, and Choppiness Index < 38.2. 
# Uses 1d trend filter (EMA34) and volume confirmation to avoid false signals. 
# Designed for 12h timeframe to target 12-37 trades/year, minimizing fee drag. 
# Works in bull markets (trend following) and bear markets (avoiding false signals via chop filter).

name = "12h_KAMA_Trend_RSI_Chop_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # KAMA (12h)
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle first er_period elements
    change = np.concatenate([np.full(er_period, np.nan), change])
    volatility = np.concatenate([np.full(er_period, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    kama = np.full_like(close, np.nan)
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, len(close)):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]

    # RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[13] = np.nanmean(gain[1:14])
    avg_loss[13] = np.nanmean(loss[1:14])
    for i in range(14, len(close)):
        if not np.isnan(avg_gain[i-1]) and not np.isnan(avg_loss[i-1]):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        else:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))

    # Choppiness Index (14)
    atr_period = 14
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = np.full_like(close, np.nan)
    atr[atr_period-1] = np.nanmean(tr[:atr_period])
    for i in range(atr_period, len(close)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    highest_high = np.full_like(close, np.nan)
    lowest_low = np.full_like(close, np.nan)
    for i in range(atr_period-1, len(close)):
        highest_high[i] = np.max(high[i-atr_period+1:i+1])
        lowest_low[i] = np.min(low[i-atr_period+1:i+1])
    chop = np.full_like(close, np.nan)
    for i in range(atr_period-1, len(close)):
        if highest_high[i] != lowest_low[i]:
            log_sum = np.sum(np.log10(atr[i-atr_period+1:i+1] / (highest_high[i] - lowest_low[i])))
            chop[i] = 100 * log_sum / np.log10(atr_period)
        else:
            chop[i] = 50.0

    # Volume spike: volume > 2.0 * 24-period average (12 days worth at 12h)
    vol_ma_24 = np.full_like(volume, np.nan)
    for i in range(24, len(volume)):
        vol_ma_24[i] = np.nanmean(volume[i-24:i])
    volume_spike = volume > 2.0 * vol_ma_24

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price > KAMA, RSI > 50, Chop < 38.2, 1d uptrend, volume spike
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                chop[i] < 38.2 and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < KAMA, RSI < 50, Chop < 38.2, 1d downtrend, volume spike
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 38.2 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price < KAMA or trend reversal or chop > 61.8 (ranging)
            if (close[i] < kama[i] or 
                close[i] < ema34_1d_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price > KAMA or trend reversal or chop > 61.8 (ranging)
            if (close[i] > kama[i] or 
                close[i] > ema34_1d_aligned[i] or 
                chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals