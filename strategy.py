#!/usr/bin/env python3
# Hypothesis: 1d KAMA trend with RSI(14) filter and volume confirmation (1.5x MA20) for mean-reversion entries.
# Long when KAMA rising, RSI<40, volume>1.5x MA20. Short when KAMA falling, RSI>60, volume>1.5x MA20.
# Exit when RSI crosses 50 (mean reversion complete) or ATR-based stoploss (2.0*ATR14).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for low trade frequency (~10-25/year)
# by requiring confluence: trend alignment + momentum extreme + volume spike.
# KAMA adapts to market noise, reducing whipsaws in choppy conditions. RSI filter avoids overbought/oversold traps.
# Volume confirmation ensures institutional participation. Works in both bull (trend continuation) and bear (mean reversion in ranges).

name = "1d_KAMA_RSI_Volume_MeanReversion_v1"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed value
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad first value
    rsi = np.concatenate([np.array([50.0]), rsi])
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # ATR(14) for stoploss
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track entry price for ATR-based stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for all indicators
        if np.isnan(kama[i]) or np.isnan(kama[i-1]) or np.isnan(rsi[i]) or \
           np.isnan(vol_ma20[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising (bullish trend), RSI<40 (oversold), volume spike
            if kama[i] > kama[i-1] and rsi[i] < 40 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: KAMA falling (bearish trend), RSI>60 (overbought), volume spike
            elif kama[i] < kama[i-1] and rsi[i] > 60 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI crosses above 50 (mean reversion) OR ATR stoploss hit
            if rsi[i] > 50 or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: RSI crosses below 50 (mean reversion) OR ATR stoploss hit
            if rsi[i] < 50 or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals