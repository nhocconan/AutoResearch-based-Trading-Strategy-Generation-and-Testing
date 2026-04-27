#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA (Kaufman Adaptive MA) with 1d RSI and 1w trend filter
# KAMA adapts to market noise - fast in trends, slow in ranges.
# RSI filters overbought/oversold conditions. Weekly trend ensures alignment with higher timeframe.
# Volume spike confirms momentum. Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d RSI(14)
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # KAMA on price (adaptive moving average)
    # Efficiency ratio: |net change| / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Calculate ER properly
    change_today = np.abs(np.diff(close, prepend=close[0]))
    change_prior = np.abs(np.diff(close, prepend=close[0]))
    
    # ER over 10 periods
    net_change = np.abs(close - np.roll(close, 10))
    total_change = np.zeros_like(close)
    for i in range(len(close)):
        start_idx = max(0, i - 9)
        total_change[i] = np.sum(np.abs(np.diff(close[start_idx:i+1])))
    
    er = np.where(total_change > 0, net_change / total_change, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: Price above KAMA (uptrend) AND RSI not overbought (<70) AND weekly uptrend + volume
        if (close[i] > kama[i] and 
            rsi_1d_aligned[i] < 70 and 
            close[i] > ema50_1w_aligned[i] and   # Weekly uptrend filter
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: Price below KAMA (downtrend) AND RSI not oversold (>30) AND weekly downtrend + volume
        elif (close[i] < kama[i] and 
              rsi_1d_aligned[i] > 30 and 
              close[i] < ema50_1w_aligned[i] and   # Weekly downtrend filter
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "12h_KAMA_1dRSI_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0