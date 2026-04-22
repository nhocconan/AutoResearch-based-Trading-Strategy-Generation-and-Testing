#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h KAMA direction + RSI + Chop regime filter
    # KAMA adapts to market noise - effective in both trending and ranging markets.
    # RSI(14) provides mean reversion signals when combined with trend filter.
    # Chop index > 61.8 identifies ranging markets where mean reversion works.
    # This combination should work in both bull (trend following) and bear (mean reversion) markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - 20 period
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else np.abs(np.diff(close, prepend=close[0]))
    # Proper volatility calculation for ER
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
    
    # Avoid division by zero
    er = np.zeros_like(close)
    for i in range(len(close)):
        if volatility[i] > 0:
            er[i] = np.abs(close[i] - close[i-1]) / volatility[i] if i > 0 else 0
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14 period)
    # Chop = 100 * log10(sum(ATR) / (log10(highest-high-lowest-low) * n))
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero and log of zero
    range_hl = highest_high - lowest_low
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    chop = np.zeros_like(close)
    for i in range(len(close)):
        if range_hl[i] > 0 and sum_atr[i] > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / (range_hl[i] * 14)) / np.log10(10)
        else:
            chop[i] = 50  # neutral value
    
    # Volume confirmation (20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(14, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend) + RSI < 40 (oversold) + Chop > 61.8 (ranging) + Volume confirmation
            if close[i] > kama[i] and rsi[i] < 40 and chop[i] > 61.8 and volume[i] > vol_ma20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend) + RSI > 60 (overbought) + Chop > 61.8 (ranging) + Volume confirmation
            elif close[i] < kama[i] and rsi[i] > 60 and chop[i] > 61.8 and volume[i] > vol_ma20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Reverse signals or Chop < 38.2 (trending market)
            if position == 1:
                if close[i] < kama[i] or rsi[i] > 60 or chop[i] < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama[i] or rsi[i] < 40 or chop[i] < 38.2:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Chop_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0