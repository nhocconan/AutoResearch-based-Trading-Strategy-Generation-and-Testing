# 1d_KAMA_Trend_RSI_Pullback_WeeklyTrend_VolumeFilter_v1
# Hypothesis: On the daily timeframe, KAMA captures adaptive trend direction. Entries occur on RSI pullbacks (30/70) in the direction of the weekly trend, confirmed by volume spikes. This strategy aims to catch trend continuations after short-term pullbacks, working in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets. Weekly trend filter reduces whipsaws, volume filter ensures institutional participation. Target: 20-50 trades/year to minimize fee drag.

#!/usr/bin/env python3
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
    
    # --- KAMA (Kaufman Adaptive Moving Average) on daily ---
    # Requires 1d data for calculation, then aligned to 1d (same timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d[9]  # start after 10 periods
    for i in range(10, len(close_1d)):
        if np.isnan(kama[i-1]):
            kama[i] = close_1d[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 1d (no change, same timeframe)
    kama_1d = kama
    
    # --- Weekly trend filter (EMA34 on 1w) ---
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # --- RSI(14) on 1d ---
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first 14 values
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # --- Volume spike (2x 20-period MA) ---
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):  # start after warmup
        # Skip if any data not ready
        if (np.isnan(kama_1d[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > KAMA (uptrend), RSI < 30 (oversold pullback), weekly EMA34 rising, volume spike
            if (close[i] > kama_1d[i] and 
                rsi[i] < 30 and 
                ema34_1w_aligned[i] > ema34_1w_aligned[i-1] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA (downtrend), RSI > 70 (overbought pullback), weekly EMA34 falling, volume spike
            elif (close[i] < kama_1d[i] and 
                  rsi[i] > 70 and 
                  ema34_1w_aligned[i] < ema34_1w_aligned[i-1] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Reverse signal or trend change
            if position == 1:
                # Exit long if price < KAMA (trend broken) or RSI > 70 (overbought)
                if close[i] < kama_1d[i] or rsi[i] > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price > KAMA (trend broken) or RSI < 30 (oversold)
                if close[i] > kama_1d[i] or rsi[i] < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_Pullback_WeeklyTrend_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0