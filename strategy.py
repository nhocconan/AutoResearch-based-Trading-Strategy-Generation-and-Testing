# 4h_KAMA_Trend_RSI_Pullback_12hTrend_Filter
# Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) as adaptive trend filter on 4h, enter on RSI pullbacks in trend direction, with 12h trend filter to avoid counter-trend trades. Designed for low trade frequency (~20-40/year) with high win rate in both bull and bear markets by avoiding whipsaws. KAMA adapts to market noise, reducing false signals during chop.

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
    
    # KAMA parameters
    fast_sc = 0.666  # 2/(2+1)
    slow_sc = 0.0645  # 2/(30+1)
    
    # Calculate KAMA on 4h close
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):  # ER needs lookback
        if i >= 10:
            direction = np.abs(close[i] - close[i-10])
            volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
            er[i] = direction / volatility if volatility != 0 else 0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss != 0, avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: above average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > vol_avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        ema_12h_val = ema_50_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price > KAMA (uptrend), RSI pulled back to 40-50, 12h trend up, volume confirmation
            if close[i] > kama_val and 40 <= rsi_val <= 50 and close[i] > ema_12h_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: price < KAMA (downtrend), RSI bounced to 50-60, 12h trend down, volume confirmation
            elif close[i] < kama_val and 50 <= rsi_val <= 60 and close[i] < ema_12h_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI > 60 (overbought) or price < KAMA (trend change)
            if rsi_val > 60 or close[i] < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 40 (oversold) or price > KAMA (trend change)
            if rsi_val < 40 or close[i] > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Trend_RSI_Pullback_12hTrend_Filter"
timeframe = "4h"
leverage = 1.0