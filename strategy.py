#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 3-period RSI mean reversion with 1-day trend filter and volatility filter.
# Enter long when RSI(3) < 15 and price > daily EMA(50), short when RSI(3) > 85 and price < daily EMA(50).
# Use ATR(14) to filter low volatility periods (ATR < 0.3 * ATRMA50).
# Exit when RSI crosses above 50 (long) or below 50 (short).
# Designed to capture mean reversion in trends while avoiding choppy markets.
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "12h_rsi3_1dema50_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # RSI(3) for mean reversion signal
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(span=3, adjust=False).mean().values
    loss_ma = pd.Series(loss).ewm(span=3, adjust=False).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR(14) for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # first period
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR > 0.3 * ATR_MA (avoid choppy low-vol periods)
        vol_filter = atr[i] > 0.3 * atr_ma[i]
        
        if position == 1:  # long position
            # Exit: RSI crosses above 50
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI crosses below 50
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: RSI extreme + EMA50 trend + volatility filter
            if vol_filter:
                if rsi[i] < 15 and close[i] > ema_50_aligned[i]:
                    # Oversold with price above EMA50: long
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 85 and close[i] < ema_50_aligned[i]:
                    # Overbought with price below EMA50: short
                    signals[i] = -0.25
                    position = -1
    
    return signals