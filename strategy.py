#!/usr/bin/env python3
"""
4h_RSI_Trend_Filter_Volume_Confirm
Hypothesis: RSI(14) extremes with trend filter (EMA50) and volume confirmation capture mean-reversion in both bull and bear markets. 
Long when RSI<30, EMA50 uptrend, volume spike. Short when RSI>70, EMA50 downtrend, volume spike.
Exit when RSI crosses 50. Uses 1d EMA50 as higher timeframe bias filter.
Target: 20-50 trades/year per symbol.
"""

name = "4h_RSI_Trend_Filter_Volume_Confirm"
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
    
    # EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_4h = close > ema_50
    downtrend_4h = close < ema_50
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = df_1d['close'].values > ema_50_1d
    downtrend_1d = df_1d['close'].values < ema_50_1d
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        rsi_val = rsi[i]
        uptrend = uptrend_4h[i]
        downtrend = downtrend_4h[i]
        uptrend_htf = uptrend_1d_aligned[i]
        downtrend_htf = downtrend_1d_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: RSI oversold, uptrend, volume confirmation
            if rsi_val < 30 and uptrend and uptrend_htf and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought, downtrend, volume confirmation
            elif rsi_val > 70 and downtrend and downtrend_htf and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses above 50 (mean reversion complete)
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI crosses below 50 (mean reversion complete)
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals