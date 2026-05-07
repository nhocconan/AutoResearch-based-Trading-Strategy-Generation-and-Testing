#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter_v1"
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
    
    # KAMA calculation (daily)
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly trend filter: EMA50
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Chop index (14)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[high[0] - low[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_sum = np.zeros(n)
    for i in range(14, n):
        atr_sum[i] = np.sum(tr[i-13:i+1])
    atr[13:] = atr_sum[13:] / 14
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    chop = np.zeros(n)
    for i in range(14, n):
        if highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    # Align weekly EMA50 to daily
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 5  # ~1 week for daily to reduce trades
    
    start_idx = max(30, 50, 14)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price above KAMA, RSI > 50, chop > 61.8 (trending)
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                chop[i] > 61.8 and 
                trend_up[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price below KAMA, RSI < 50, chop > 61.8 (trending)
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  chop[i] > 61.8 and 
                  trend_down[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price crosses below KAMA or chop < 38.2 (ranging)
            if close[i] < kama[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above KAMA or chop < 38.2 (ranging)
            if close[i] > kama[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Using daily KAMA as trend filter with RSI momentum and Chop index regime filter.
# Long when price > KAMA, RSI > 50, chop > 61.8 (trending up), short when price < KAMA, RSI < 50, chop > 61.8 (trending down).
# Weekly EMA50 ensures alignment with higher timeframe trend. Chop index filters out ranging markets (chop < 38.2).
# This strategy should work in both bull and bear markets by following the weekly trend while using daily momentum.
# Position size 0.25 manages drawdown, cooldown of 5 days prevents overtrading (~70 trades/year max).