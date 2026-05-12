#!/usr/bin/env python3
"""
1D_KAMA_TREND_RSI_WITH_VOLUME_CONFIRMATION
Hypothesis: Use 1d KAMA for trend direction, RSI for pullback entry, and volume spike for confirmation.
KAMA adapts to market noise, reducing whipsaw in sideways markets. RSI identifies overextended pullbacks
within the trend. Volume spike ensures institutional participation. Works in bull markets (buy pullbacks
in uptrend) and bear markets (sell rallies in downtrend). Target: 15-25 trades/year (60-100 total) to
stay within 1d limits and minimize fee drag.
"""
name = "1D_KAMA_TREND_RSI_WITH_VOLUME_CONFIRMATION"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA trend: 1d close, fast=2, slow=30
    close_s = pd.Series(close)
    change = abs(close_s.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [close[0]]
    for i in range(1, n):
        kama.append(kama[-1] + sc.iloc[i] * (close[i] - kama[-1]))
    kama = np.array(kama)
    
    # RSI(14) for pullback
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup for volatility and volume MA
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend (close > KAMA) + RSI pullback (RSI < 30) + volume spike
            if close[i] > kama[i] and rsi[i] < 30 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (close < KAMA) + RSI overextended (RSI > 70) + volume spike
            elif close[i] < kama[i] and rsi[i] > 70 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal (close < KAMA) OR RSI overbought (RSI > 70)
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal (close > KAMA) OR RSI oversold (RSI < 30)
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals