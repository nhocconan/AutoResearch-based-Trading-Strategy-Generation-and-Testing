#!/usr/bin/env python3
# 4h_RSI_Trend_Filter_Volume_Confirm
# Hypothesis: RSI(14) with EMA(50) trend filter and volume confirmation provides high-probability entries in both bull and bear markets.
# Long when RSI < 40 and price > EMA(50) with volume > 1.5x average.
# Short when RSI > 60 and price < EMA(50) with volume > 1.5x average.
# Exit when RSI returns to neutral range (40-60).
# Uses 4h timeframe for lower trade frequency (target: 20-40 trades/year) to minimize fee drag.
# Includes ATR-based stoploss via signal=0 when adverse move exceeds 2x ATR.

name = "4h_RSI_Trend_Filter_Volume_Confirm"
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
    
    # RSI(14)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    loss_ma = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA(50) for trend filter
    ema_period = 50
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    
    # ATR(14) for stoploss
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(rsi_period, ema_period, atr_period, 20)
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long entry: RSI oversold, price above EMA, volume confirmation
            if rsi[i] < 40 and close[i] > ema[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI overbought, price below EMA, volume confirmation
            elif rsi[i] > 60 and close[i] < ema[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral or stoploss hit
            if rsi[i] >= 40 or close[i] < ema[i] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral or stoploss hit
            if rsi[i] <= 60 or close[i] > ema[i] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals