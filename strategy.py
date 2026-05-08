#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA with RSI and chop filter on 1d timeframe.
# Uses KAMA(10) for trend direction, RSI(14) for momentum, and Chop(14) for regime.
# Long when KAMA up, RSI > 50, Chop < 38.2 (trending). Short when KAMA down, RSI < 50, Chop < 38.2.
# In chop (Chop > 61.8), mean revert at Bollinger Bands (20,2).
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in both bull (trend follow) and bear (mean revert in chop).

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # KAMA(10) on 1d
    def kama(close, period=10):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
        kama = np.full_like(close, np.nan)
        kama[period] = close[period]
        for i in range(period + 1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, 10)
    kama_up = kama_vals > np.roll(kama_vals, 1)
    kama_up = np.where(np.isnan(kama_up), False, kama_up)
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[:14])
    avg_loss[14] = np.mean(loss[:14])
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])
    
    # Chop(14)
    def chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        for i in range(period, len(close)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        sum_atr = np.sum(atr, axis=1)
        max_high = np.maximum.accumulate(high)
        min_low = np.minimum.accumulate(low)
        range_hl = max_high - min_low
        chop_val = 100 * np.log10(sum_atr / range_hl) / np.log10(period)
        return chop_val
    
    chop_vals = chop(high, low, close, 14)
    
    # Bollinger Bands (20,2) for mean reversion in chop
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # 1w EMA(10) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_10_1w = close_1w_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    trend_1w_up = ema_10_1w[1:] > ema_10_1w[:-1]
    trend_1w_up = np.concatenate([[False], trend_1w_up])
    
    # Align 1w trend to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for BB
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_up[i]) or np.isnan(rsi[i]) or np.isnan(chop_vals[i]) or
            np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(trend_1w_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trending regime: Chop < 38.2
            if chop_vals[i] < 38.2:
                # Long: KAMA up, RSI > 50
                if kama_up[i] and rsi[i] > 50:
                    signals[i] = 0.25
                    position = 1
                # Short: KAMA down, RSI < 50
                elif not kama_up[i] and rsi[i] < 50:
                    signals[i] = -0.25
                    position = -1
            # Choppy regime: Chop > 61.8, mean revert at Bollinger Bands
            elif chop_vals[i] > 61.8:
                # Long at lower BB
                if close[i] <= lower_bb[i]:
                    signals[i] = 0.25
                    position = 1
                # Short at upper BB
                elif close[i] >= upper_bb[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: opposite signal or chop exit
            if chop_vals[i] < 38.2:
                # Trending: exit when KAMA down or RSI < 50
                if not kama_up[i] or rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Choppy: exit when price crosses SMA
                if close[i] >= sma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit: opposite signal or chop exit
            if chop_vals[i] < 38.2:
                # Trending: exit when KAMA up or RSI > 50
                if kama_up[i] or rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Choppy: exit when price crosses SMA
                if close[i] <= sma_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals