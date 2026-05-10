#12h_KAMA_Trend_RSI_Pullback
# Hypothesis: Use KAMA to detect trend direction on 12h, enter on RSI pullbacks in trend direction with volume confirmation.
# KAMA adapts to market noise, reducing whipsaws in sideways markets. RSI pullbacks provide better entry prices.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 20-30 trades/year to stay under fee drag limits.

name = "12h_KAMA_Trend_RSI_Pullback"
timeframe = "12h"
leverage = 1.0

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
    
    # KAMA trend on 12h (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d trend filter: EMA(34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # RSI(14) for pullback entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.3 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI pullback (<40), volume confirmation
            if close[i] > kama[i] and rsi[i] < 40 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI rally (>60), volume confirmation
            elif close[i] < kama[i] and rsi[i] > 60 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA OR RSI overbought (>70)
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA OR RSI oversold (<30)
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals