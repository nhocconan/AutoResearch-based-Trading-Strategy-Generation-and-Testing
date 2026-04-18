#!/usr/bin/env python3
"""
4h_KAMA_Reversal_With_Volume_and_Trend_Filter
Hypothesis: KAMA direction combined with RSI extremes and volume spikes provides reliable mean-reversion entries in both bull and bear markets. KAMA adapts to market noise, reducing false signals during choppy periods while capturing reversals at extremes. Trend filter (12h EMA34) ensures alignment with higher timeframe momentum. Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = [close_s.iloc[0]]
    for i in range(1, len(close_s)):
        kama.append(kama[-1] + sc.iloc[i] * (close_s.iloc[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI(14)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Trend filter: 12h EMA34
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        ema34 = ema_34_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price below KAMA (oversold) + RSI oversold + volume spike + uptrend
            if price < kama_val and rsi_val < 30 and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price above KAMA (overbought) + RSI overbought + volume spike + downtrend
            elif price > kama_val and rsi_val > 70 and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses above KAMA OR RSI overbought
            if price > kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses below KAMA OR RSI oversold
            if price < kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Reversal_With_Volume_and_Trend_Filter"
timeframe = "4h"
leverage = 1.0