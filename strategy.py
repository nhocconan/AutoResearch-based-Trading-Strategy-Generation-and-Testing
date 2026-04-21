#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_RSI_and_Chop_Filter
Hypothesis: KAMA adapts to market noise, providing reliable trend direction. 
Combine with RSI for momentum and Choppiness Index to filter ranging markets.
Enter long when KAMA upward, RSI>55, and trending market (CHOP<40). 
Enter short when KAMA downward, RSI<45, and trending market.
Exit on opposite signal. Uses 1D trend filter for higher timeframe bias.
Works in bull markets by riding uptrends and in bear markets by catching bounces.
Target: 15-25 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1D EMA34 for trend filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === KAMA (12h) ===
    close = prices['close'].values
    direction = np.abs(np.diff(close, k=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of abs changes
    # Handle first 10 values
    direction = np.concatenate([np.full(10, np.nan), direction])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (12h, 14-period) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    avg_gain[14] = np.nanmean(gain[1:15])
    avg_loss[14] = np.nanmean(loss[1:15])
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index (12h, 14-period) ===
    atr = np.full_like(close, np.nan)
    tr1 = np.abs(prices['high'].values - prices['low'].values)
    tr2 = np.abs(prices['high'].values - np.roll(close, 1))
    tr3 = np.abs(prices['low'].values - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr_period = 14
    atr[atr_period-1] = np.nanmean(tr[1:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    max_high = np.full_like(close, np.nan)
    min_low = np.full_like(close, np.nan)
    for i in range(atr_period-1, n):
        max_high[i] = np.max(prices['high'].values[i-atr_period+1:i+1])
        min_low[i] = np.min(prices['low'].values[i-atr_period+1:i+1])
    chop = np.where((max_high - min_low) != 0, 
                    100 * np.log10(np.sum(tr[i-atr_period+1:i+1]) / (max_high - min_low)) / np.log10(atr_period),
                    50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema_trend = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: KAMA up, RSI>55, trending market (CHOP<40), above 1D EMA
            if (kama_val > kama[i-1] and
                rsi_val > 55 and
                chop_val < 40 and
                price_close > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI<45, trending market (CHOP<40), below 1D EMA
            elif (kama_val < kama[i-1] and
                  rsi_val < 45 and
                  chop_val < 40 and
                  price_close < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit on opposite signal
            if position == 1 and (kama_val < kama[i-1] or rsi_val < 45):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (kama_val > kama[i-1] or rsi_val > 55):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_KAMA_Trend_With_RSI_and_Chop_Filter"
timeframe = "12h"
leverage = 1.0