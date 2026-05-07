#!/usr/bin/env python3
name = "1d_KAMA_Direction_RSI_ChopFilter"
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
    
    # KAMA parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros(n)
    for i in range(er_period, n):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI calculation
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.zeros(n)
    rs[rsi_period:] = avg_gain[rsi_period:] / np.where(avg_loss[rsi_period:] == 0, 1, avg_loss[rsi_period:])
    rsi = np.zeros(n)
    rsi[rsi_period:] = 100 - (100 / (1 + rs[rsi_period:]))
    
    # Choppy Market Index calculation
    chop_period = 14
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[0], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_sum = np.zeros(n)
    for i in range(chop_period, n):
        atr_sum[i] = np.sum(tr[i-chop_period+1:i+1])
    
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(chop_period-1, n):
        highest_high[i] = np.max(high[i-chop_period+1:i+1])
        lowest_low[i] = np.min(low[i-chop_period+1:i+1])
    
    chop = np.zeros(n)
    for i in range(chop_period-1, n):
        if highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(chop_period)
        else:
            chop[i] = 50
    
    # Weekly trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    weekly_trend_up = close > ema_21_1w_aligned
    weekly_trend_down = close < ema_21_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # 2 days cooldown
    
    start_idx = max(er_period, rsi_period, chop_period) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema_21_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: KAMA up, RSI > 50, Chop < 61.8 (trending), Weekly uptrend
            if (close[i] > kama[i] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and 
                weekly_trend_up[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: KAMA down, RSI < 50, Chop < 61.8 (trending), Weekly downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and 
                  weekly_trend_down[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: KAMA down or Chop > 61.8 (choppy) or Weekly trend changes
            if (close[i] < kama[i] or 
                chop[i] > 61.8 or 
                not weekly_trend_up[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA up or Chop > 61.8 (choppy) or Weekly trend changes
            if (close[i] > kama[i] or 
                chop[i] > 61.8 or 
                not weekly_trend_down[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On 1d timeframe, KAMA direction filters noise, RSI >50/<50 confirms momentum, Chop <61.8 ensures trending market (not ranging), and weekly EMA21 aligns with higher timeframe trend. This combination avoids whipsaws in ranging markets while capturing trends in both bull and bear markets. Cooldown prevents overtrading. Target: 30-100 trades over 4 years (7-25/year). Works in bull markets (KAMA up in uptrend) and bear markets (KAMA down in downtrend). Uses discrete position sizing (0.25) to balance risk and minimize fee churn.