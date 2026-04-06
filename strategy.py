#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day KAMA (Kaufman Adaptive Moving Average) with RSI(14) and Choppiness Index regime filter
# Long when KAMA slope > 0, RSI(14) > 50, and Choppiness Index < 40 (trending market)
# Short when KAMA slope < 0, RSI(14) < 50, and Choppiness Index < 40 (trending market)
# Exit when RSI crosses 50 or Choppiness Index > 60 (ranging market)
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses weekly trend filter from 1w EMA(50) to avoid counter-trend trades
# Works in bull/bear by following weekly trend direction and using RSI for momentum
# Target: 50-100 trades over 4 years (12-25/year)

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # KAMA (Kaufman Adaptive Moving Average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    change[0:10] = 0
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if hasattr(np, 'sum') else None
    # Manual volatility calculation for 10-period rolling sum
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.where(tr_sum > 0, 100 * np.log10((hh - ll) / tr_sum) / np.log10(14), 50)
    
    # ATR(14) for stoploss
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses below 50 or choppy market (chop > 60)
            elif rsi[i] < 50 or chop[i] > 60:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses above 50 or choppy market (chop > 60)
            elif rsi[i] > 50 or chop[i] > 60:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend alignment and momentum
            # Long: KAMA rising, RSI > 50, trending market (chop < 40), and above weekly EMA
            if (kama[i] > kama[i-1] and  # KAMA slope positive
                rsi[i] > 50 and
                chop[i] < 40 and
                close[i] > ema_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: KAMA falling, RSI < 50, trending market (chop < 40), and below weekly EMA
            elif (kama[i] < kama[i-1] and  # KAMA slope negative
                  rsi[i] < 50 and
                  chop[i] < 40 and
                  close[i] < ema_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals