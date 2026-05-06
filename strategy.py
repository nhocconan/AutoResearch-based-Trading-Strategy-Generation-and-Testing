#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h/1d trend filter and volume confirmation
# Uses 4h EMA50 for trend direction and 1d RSI for overbought/oversold extremes
# Entry: RSI(14) < 30 in uptrend (price > 4h EMA50) or RSI > 70 in downtrend (price < 4h EMA50)
# Volume filter: current volume > 1.5x 20-bar average to ensure participation
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Fixed position size 0.20 to limit drawdown and reduce trade frequency
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Works in bull/bear: mean reversion works in ranges, trend filter avoids counter-trend traps

name = "1h_RSI14_4hEMA50_1dRSI_MeanRev_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1d RSI(14) for overbought/oversold levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1h RSI(14) for entry signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Volume filter: >1.5x 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(rsi_14[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: RSI oversold (<30) AND uptrend (price > 4h EMA50) AND volume
            if rsi_14[i] < 30 and close[i] > ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short signal: RSI overbought (>70) AND downtrend (price < 4h EMA50) AND volume
            elif rsi_14[i] > 70 and close[i] < ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI returns to neutral (>50) or trend fails
            if rsi_14[i] > 50 or close[i] <= ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI returns to neutral (<50) or trend fails
            if rsi_14[i] < 50 or close[i] >= ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals