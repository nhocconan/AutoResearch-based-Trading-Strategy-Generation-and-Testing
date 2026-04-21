#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA direction + RSI + chop filter
# Uses 12h KAMA to determine trend direction, 1d RSI for overbought/oversold levels,
# and 1d Choppiness Index to filter ranging markets. Long when KAMA up, RSI < 50, and CHOP > 61.8 (ranging).
# Short when KAMA down, RSI > 50, and CHOP > 61.8. This mean-reversion strategy works in both bull and bear
# markets by fading moves in ranging conditions while avoiding strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    avg_gain = wilder_smooth(gain, 14)
    avg_loss = wilder_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # Calculate 1d Choppiness Index(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR and sum of ranges
    atr = wilder_smooth(tr, 14)
    sum_tr = np.nancumsum(atr) - np.nancumsum(np.where(np.arange(len(atr)) < 14, atr, 0))
    sum_tr = np.where(np.arange(len(atr)) >= 13, sum_tr, np.nan)
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum_tr / (hh - ll)) / log10(14)
    range_hl = hh - ll
    chop = np.where(range_hl > 0, 100 * np.log10(sum_tr / range_hl) / np.log10(14), 50)
    
    # Align RSI and Chop to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 12h KAMA(10, 2, 30)
    close = prices['close'].values
    direction = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.nancumsum(np.abs(np.diff(close, prepend=close[0]))) - np.nancumsum(np.where(np.arange(len(close)) < 1, np.abs(np.diff(close, prepend=close[0])), 0))
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full_like(close, np.nan, dtype=float)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.where(kama > np.roll(kama, 1), 1, -1)
    kama_dir[0] = 1
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(30, n):
        if np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or np.isnan(kama_dir[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade in ranging markets (CHOP > 61.8)
        if chop_aligned[i] <= 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up and RSI < 50
            if kama_dir[i] == 1 and rsi_aligned[i] < 50:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down and RSI > 50
            elif kama_dir[i] == -1 and rsi_aligned[i] > 50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when KAMA direction changes or RSI reverts to 50
            exit_signal = False
            if position == 1:  # long
                if kama_dir[i] == -1 or rsi_aligned[i] >= 50:
                    exit_signal = True
            elif position == -1:  # short
                if kama_dir[i] == 1 or rsi_aligned[i] <= 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_KAMA_RSI_Chop_MeanReversion"
timeframe = "12h"
leverage = 1.0