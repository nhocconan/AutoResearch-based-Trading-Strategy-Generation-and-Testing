#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day KAMA trend direction with RSI mean reversion and volume filter
# Long when 1-day KAMA is rising AND RSI(14) < 40 AND volume > 1.5x 20-period average
# Short when 1-day KAMA is falling AND RSI(14) > 60 AND volume > 1.5x 20-period average
# Uses daily KAMA for trend filter, RSI for mean reversion entries, volume for confirmation
# Designed to work in bull markets via pullbacks in uptrend and in bear markets via bounces in downtrend
# Target: 20-30 trades per year (80-120 over 4 years) with 0.25 position sizing

name = "4h_1dKAMA_RSI_Volume_v1"
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
    
    # Calculate 1-day KAMA (adaptive moving average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate efficiency ratio (ER)
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0) if len(close_1d) > 1 else np.array([0])
    # Simplified ER calculation for arrays
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            ch = np.abs(close_1d[i] - close_1d[i-10])
            vol = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            er[i] = ch / vol if vol != 0 else 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # KAMA trend: rising if current > previous, falling if current < previous
    kama_rising = kama_aligned > np.roll(kama_aligned, 1)
    kama_falling = kama_aligned < np.roll(kama_aligned, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising AND RSI < 40 (oversold in uptrend) with volume confirmation
            if kama_rising[i] and rsi[i] < 40 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND RSI > 60 (overbought in downtrend) with volume confirmation
            elif kama_falling[i] and rsi[i] > 60 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down OR RSI > 70 (overbought)
            if not kama_rising[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up OR RSI < 30 (oversold)
            if not kama_falling[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals