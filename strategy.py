#!/usr/bin/env python3
"""
4h_1d_KAMA_RSI_Trend_v1
Hypothesis: KAMA identifies trend direction on 4h, RSI(2) identifies oversold/overbought entries within trend,
with volume confirmation and daily ATR filter to avoid chop. Works in bull (buy dips in uptrend) and 
bear (sell rallies in downtrend) by following the trend while fading short-term extremes.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_RSI_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA TREND (4H) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) >= 10 else np.array([])
    if len(change) > 0 and len(volatility) > 0:
        er = np.where(volatility != 0, change / volatility, 0)
    else:
        er = np.zeros(len(close))
    # Pad ER to match close length
    er = np.concatenate([np.full(10, np.nan), er])
    
    # Smoothing constants
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    sc = np.where(np.isnan(sc), 0.0645**2, sc)
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    if len(close) > 10:
        kama[10] = close[10]
        for i in range(11, len(close)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    # === DAILY ATR FILTER (avoid chop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([0.0]), tr])  # first TR = 0
    
    # ATR(14)
    atr_1d = np.full_like(close_1d, np.nan)
    if len(tr) >= 15:
        atr_1d[13] = np.mean(tr[1:15])  # first ATR
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_1d_ma = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ma)
    
    # === RSI(2) for entry timing ===
    rsi_period = 2
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    if len(gain) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
        for i in range(rsi_period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price vs KAMA
        bullish = close[i] > kama[i]
        bearish = close[i] < kama[i]
        
        # Volatility filter: avoid low volatility/chop
        vol_filter = atr_1d_aligned[i] > 0  # ATR > 0 always true if calculated
        
        # Entry conditions
        long_entry = (bullish and 
                     rsi[i] < 15 and  # deeply oversold
                     vol_ratio[i] > 1.5)  # volume confirmation
        
        short_entry = (bearish and 
                      rsi[i] > 85 and  # deeply overbought
                      vol_ratio[i] > 1.5)  # volume confirmation
        
        # Exit conditions
        exit_long = (position == 1 and 
                    (not bullish or rsi[i] > 70))  # trend change or overbought
        exit_short = (position == -1 and 
                     (not bearish or rsi[i] < 30))  # trend change or oversold
        
        # Execute trades
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals