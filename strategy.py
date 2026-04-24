#!/usr/bin/env python3
"""
Hypothesis: 4h KAMA trend with RSI mean reversion and chop filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d KAMA for trend filter (price above/below KAMA defines trend).
- Entry: Long when price > 4h KAMA AND RSI(14) < 30 AND chop > 61.8 (range regime);
         Short when price < 4h KAMA AND RSI(14) > 70 AND chop > 61.8 (range regime).
- Exit: ATR-based trailing stop (2.0 * ATR(14)) from highest high/lowest low since entry.
- Signal size: 0.25 discrete to control fee drag.
- Uses KAMA for adaptive trend, RSI for mean reversion in chop, chop filter to avoid trending markets.
- Designed to work in both bull (longs on dips) and bear (shorts on rallies) ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h KAMA for trend filter
    # KAMA parameters: ER period=10, fastest EMA=2, slowest EMA=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1d data for KAMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d KAMA
    change_1d = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_1d = np.abs(np.diff(close_1d))
    volatility_sum_1d = pd.Series(volatility_1d).rolling(window=10, min_periods=1).sum().values
    er_1d = np.where(volatility_sum_1d > 0, change_1d / volatility_sum_1d, 0)
    sc_1d = (er_1d * (2/2 - 2/30) + 2/30) ** 2
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) for 4h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Chopiness Index(14) for regime filter
    # Chop = 100 * log10(sum(TR) / (ATR * N)) / log10(N)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])  # first TR is high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr_times_n = atr * 14
    chop = np.where(atr_times_n > 0, 100 * np.log10(tr_sum / atr_times_n) / np.log10(14), 50)
    
    # Calculate ATR(14) for stoploss
    atr14 = atr  # reuse from chop calculation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(14, 10)  # RSI and chop need 14, KAMA needs 10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr14[i]
        
        if position == 0:
            # Check for entry signals in choppy market (chop > 61.8 = ranging)
            if chop[i] > 61.8:
                # Long: Price above KAMA (uptrend) AND RSI oversold (<30)
                if curr_close > kama[i] and curr_close > kama_1d_aligned[i] and rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
                # Short: Price below KAMA (downtrend) AND RSI overbought (>70)
                elif curr_close < kama[i] and curr_close < kama_1d_aligned[i] and rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
        elif position == 1:
            # Long position: update highest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.0 * ATR below highest high since entry
            stoploss = highest_since_entry - 2.0 * curr_atr
            if curr_close < stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.0 * ATR above lowest low since entry
            stoploss = lowest_since_entry + 2.0 * curr_atr
            if curr_close > stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Chop_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0