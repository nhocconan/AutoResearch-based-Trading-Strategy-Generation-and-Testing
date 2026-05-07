#!/usr/bin/env python3
# 1D_KAMA_RSI_CHOP_FILTER
# Hypothesis: Daily KAMA direction provides the primary trend signal, confirmed by RSI for momentum and Choppiness Index for regime filtering.
# This combination aims to capture sustained moves in both bull and bear markets while avoiding whipsaws in sideways conditions.
# Target: 20-60 trades over 4 years (5-15/year) with position size 0.25.

#!/usr/bin/env python3
name = "1D_KAMA_RSI_CHOP_FILTER"
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
    
    # KAMA: Kaufman Adaptive Moving Average
    # ER (Efficiency Ratio) = |change| / volatility
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will compute properly below
    # Recompute volatility as sum of absolute changes over ER period
    er_period = 10
    fast_sc = 2 / (2 + 1)   # for EMA 2
    slow_sc = 2 / (30 + 1)  # for EMA 30
    
    change_arr = np.abs(np.diff(close, prepend=close[0]))
    # Volatility: sum of absolute changes over er_period
    volatility_arr = np.nansum(np.abs(np.diff(close)).reshape(-1, 1), axis=0)  # placeholder, will fix
    # Correct volatility calculation: rolling sum of absolute price changes
    abs_changes = np.abs(np.diff(close, prepend=close[0]))
    volatility_sum = np.convolve(abs_changes, np.ones(er_period), mode='same')
    # Avoid division by zero
    volatility_sum = np.where(volatility_sum == 0, 1, volatility_sum)
    er = np.divide(change_arr, volatility_sum, out=np.zeros_like(change_arr), where=volatility_sum!=0)
    # Smooth ER
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Use Wilder's smoothing (alpha = 1/period)
    rsi_period = 14
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:rsi_period+1] = 50  # neutral before enough data
    
    # Choppiness Index (14)
    chop_period = 14
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    # Sum of TR over chop_period
    tr_sum = np.convolve(tr, np.ones(chop_period), mode='same')
    # Highest high and lowest low over chop_period
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(n):
        start = max(0, i - chop_period + 1)
        highest_high[i] = np.max(high[start:i+1])
        lowest_low[i] = np.min(low[start:i+1])
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    hh_ll = np.where(hh_ll == 0, 1, hh_ll)
    chop = 100 * np.log10(tr_sum / hh_ll) / np.log10(chop_period)
    chop[:chop_period-1] = 50  # neutral before enough data
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 21)  # KAMA, RSI, CHOP, weekly warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_21_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_uptrend = close[i] > ema_21_1w_aligned[i]
        weekly_downtrend = close[i] < ema_21_1w_aligned[i]
        
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI conditions: avoid extremes, favor momentum
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Chop filter: only trade when not too choppy (trending market)
        # Chop > 61.8 = ranging, Chop < 38.2 = trending
        not_choppy = chop[i] < 61.8  # allow some chop, avoid strong ranging
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI bullish, not overbought, weekly uptrend, not choppy
            if (price_above_kama and rsi_bullish and rsi_not_overbought and 
                weekly_uptrend and not_choppy):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI bearish, not oversold, weekly downtrend, not choppy
            elif (price_below_kama and rsi_bearish and rsi_not_oversold and 
                  weekly_downtrend and not_choppy):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price below KAMA OR RSI turns bearish OR weekly trend reverses OR choppy
            if (not price_above_kama or not rsi_bullish or not weekly_uptrend or chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price above KAMA OR RSI turns bullish OR weekly trend reverses OR choppy
            if (not price_below_kama or not rsi_bearish or not weekly_downtrend or chop[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals