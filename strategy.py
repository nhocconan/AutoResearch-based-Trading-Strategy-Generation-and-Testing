#!/usr/bin/env python3
"""
4H_RSI_MeanReversion_With_TrendFilter
Hypothesis: RSI mean reversion works best when filtered by trend direction (using 1d EMA) and volatility regime (using 1d ATR ratio). In bull markets, we take long signals when RSI is oversold in an uptrend; in bear markets, we take short signals when RSI is overbought in a downtrend. Volatility filter ensures we only trade when volatility is normal-to-high, avoiding chop.
"""

name = "4H_RSI_MeanReversion_With_TrendFilter"
timeframe = "4h"
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
    
    # Calculate RSI (14) on 4h
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1d data for trend filter (EMA34) and volatility filter (ATR ratio)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema34_1d = np.zeros_like(close_1d)
    ema34_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema34_1d[i] = (close_1d[i] * 2 / (34 + 1)) + (ema34_1d[i-1] * (33 / (34 + 1)))
    
    # 1d ATR for volatility filter
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    
    atr_period = 14
    atr_1d = np.zeros_like(close_1d)
    atr_1d[:atr_period] = np.mean(tr_1d[:atr_period])
    for i in range(atr_period, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * (atr_period-1) + tr_1d[i]) / atr_period
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime)
    atr_ma_50 = np.zeros_like(atr_1d)
    for i in range(49, len(atr_1d)):
        atr_ma_50[i] = np.mean(atr_1d[i-49:i+1])
    
    atr_ratio = np.where(atr_ma_50 > 0, atr_1d / atr_ma_50, 1.0)
    
    # Align 1d indicators to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volatility filter: ATR ratio between 0.5 and 2.0 (avoid low volatility chop)
        vol_filter = (atr_ratio_aligned[i] >= 0.5) and (atr_ratio_aligned[i] <= 2.0)
        
        if position == 0:
            # LONG: RSI oversold (<30) in uptrend with adequate volatility
            if (rsi[i] < 30 and uptrend and vol_filter):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) in downtrend with adequate volatility
            elif (rsi[i] > 70 and downtrend and vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI overbought (>70) or trend change
            if (rsi[i] > 70 or not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI oversold (<30) or trend change
            if (rsi[i] < 30 or not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals