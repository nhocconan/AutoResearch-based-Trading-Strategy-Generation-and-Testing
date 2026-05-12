#!/usr/bin/env python3
# 4H_KAMA_21_RSI_14_CHOP_14_12H_TREND
# Hypothesis: On 4h timeframe, use KAMA(21) for trend direction, RSI(14) for momentum,
# and Choppiness Index(14) for regime filtering. Enter long when KAMA is rising,
# RSI crosses above 50, and market is trending (CHOP < 38.2). Enter short when
# KAMA is falling, RSI crosses below 50, and market is trending. Exit when RSI
# reaches opposite extreme (70 for long, 30 for short) or trend changes.
# Uses 12h EMA50 as higher timeframe trend filter to avoid counter-trend trades.
# Designed to work in both bull and bear markets by adapting to trending regimes.

name = "4H_KAMA_21_RSI_14_CHOP_14_12H_TREND"
timeframe = "4h"
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
    
    # KAMA (Kaufman Adaptive Moving Average) - 21 period
    # ER = |Change| / Sum(|ΔClose|) over period
    change = np.abs(np.diff(close, prepend=close[0]))
    dir = np.abs(np.diff(close, k=21, prepend=close[:21]))
    vol = np.convolve(change, np.ones(21), 'same')
    vol[:10] = np.nancumsum(change[:11])[::-1][:10]  # forward fill start
    vol[-10:] = np.nancumsum(change[::-1])[:10][::-1]  # backward fill end
    er = np.where(vol != 0, dir / vol, 0)
    sc = (er * 0.6645 + 0.0645) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14 period) - Wilder's smoothing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14 period)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.convolve(tr, np.ones(14)/14, 'same')
    atr[:6] = np.nan
    atr[-7:] = np.nan
    # True Range sum over 14 periods
    tr_sum = np.convolve(tr, np.ones(14), 'same')
    tr_sum[:13] = np.nan
    tr_sum[-13:] = np.nan
    # Max and min close over 14 periods
    max_close = np.zeros_like(close)
    min_close = np.zeros_like(close)
    for i in range(n):
        start = max(0, i-13)
        end = i+1
        max_close[i] = np.max(close[start:end])
        min_close[i] = np.min(close[start:end])
    chop = 100 * np.log10(tr_sum / (max_close - min_close)) / np.log10(14)
    chop = np.where((max_close - min_close) != 0, chop, 50)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation (20-period average)
    vol_ma = np.convolve(volume, np.ones(20)/20, 'same')
    vol_ma[:19] = np.nan
    vol_ma[-19:] = np.nan
    vol_spike = volume > vol_ma * 1.5  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising, RSI > 50, trending market (CHOP < 38.2), volume spike, 12h uptrend
            if (kama[i] > kama[i-1] and rsi[i] > 50 and chop[i] < 38.2 and 
                vol_spike[i] and close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI < 50, trending market (CHOP < 38.2), volume spike, 12h downtrend
            elif (kama[i] < kama[i-1] and rsi[i] < 50 and chop[i] < 38.2 and 
                  vol_spike[i] and close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 70 or trend change (KAMA falling or CHOP > 61.8)
            if rsi[i] > 70 or kama[i] < kama[i-1] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 30 or trend change (KAMA rising or CHOP > 61.8)
            if rsi[i] < 30 or kama[i] > kama[i-1] or chop[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals