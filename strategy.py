#!/usr/bin/env python3
name = "1d_KAMA_Adaptive_Trend_With_RSI_and_Chop"
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
    
    # Weekly trend filter (1w EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    trend_up = close > ema_200_1w_aligned
    trend_down = close < ema_200_1w_aligned
    
    # KAMA on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) on daily close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) on daily high/low/close
    atr14 = np.full(n, np.nan)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    for i in range(1, n):
        tr[i] = np.maximum(high[i] - low[i], np.maximum(np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1])))
    for i in range(14, n):
        atr14[i] = np.sum(tr[i-13:i+1]) / 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(14, n):
        highest_high[i] = np.max(high[i-13:i+1])
        lowest_low[i] = np.min(low[i-13:i+1])
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if atr14[i] > 0:
            chop[i] = 100 * np.log10(highest_high[i] - lowest_low[i]) / np.log10(14) / atr14[i]
        else:
            chop[i] = 50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # 3 days to prevent overtrading
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: KAMA up + RSI > 50 + Chop < 61.8 (trending) + weekly uptrend
            if (kama[i] > kama[i-1] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and 
                trending_up):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: KAMA down + RSI < 50 + Chop < 61.8 (trending) + weekly downtrend
            elif (kama[i] < kama[i-1] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and 
                  trending_down):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: KAMA reverses down or chop > 61.8 (choppy) or weekly trend changes
            if (kama[i] < kama[i-1] or 
                chop[i] > 61.8 or 
                not trending_up):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA reverses up or chop > 61.8 (choppy) or weekly trend changes
            if (kama[i] > kama[i-1] or 
                chop[i] > 61.8 or 
                not trending_down):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: On daily timeframe, KAMA adapts to market efficiency to identify trend direction, filtered by RSI for momentum strength and Chop < 61.8 to ensure trending (not choppy) conditions. Weekly EMA200 provides higher timeframe trend alignment. This combination captures sustained trends in both bull and bear markets while avoiding false signals in sideways chop. Entry requires KAMA direction + RSI >/< 50 + Chop < 61.8 + weekly trend alignment. Exit on KAMA reversal, chop > 61.8, or weekly trend change. Position size 0.25 balances capture and risk. Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag while capturing significant moves. Works in bull markets (KAMA up in uptrend) and bear markets (KAMA down in downtrend).