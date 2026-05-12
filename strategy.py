#!/usr/bin/env python3
name = "4h_KAMA_Direction_RSI_12hTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA on 4h close
    close_s = pd.Series(close)
    change = close_s.diff().abs()
    volatility = close_s.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) on 4h close
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # 12h EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # ensure sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if KAMA or EMA not ready
        if np.isnan(kama[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA + RSI > 50 + 12h EMA50 uptrend + volume confirmation
            if (close[i] > kama[i]) and (rsi[i] > 50) and (close[i] > ema_50_12h_aligned[i]) and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + RSI < 50 + 12h EMA50 downtrend + volume confirmation
            elif (close[i] < kama[i]) and (rsi[i] < 50) and (close[i] < ema_50_12h_aligned[i]) and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below KAMA or RSI < 40
            if (close[i] < kama[i]) or (rsi[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above KAMA or RSI > 60
            if (close[i] > kama[i]) or (rsi[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals