#3/0.110
#!/usr/bin/env python3
"""
12h KAMA + RSI + Chop Regime Filter (12h)
Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely,
in ranging markets it stays flat. Combined with RSI momentum and Chop filter to avoid
false signals in low volatility regimes. Designed for low trade frequency (<30/year)
to minimize fee drag while capturing sustained moves in both bull and bear markets.
12h timeframe reduces trade frequency to avoid fee drag, using 1d HTF for regime confirmation.
"""
name = "12h_KAMA_RSI_ChopFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA (Adaptive Moving Average) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros(n)
    for i in range(1, n):
        if np.sum(volatility[max(0, i-9):i+1]) > 0:
            er[i] = change[i] / np.sum(volatility[max(0, i-9):i+1])
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI (14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Chop Index (14) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (highest_high - lowest_low)) / np.log10(14)
    
    # === Volume Spike (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # === 1d HTF Trend (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend from 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # LONG: Price above KAMA + RSI > 50 + Chop < 61.8 (trending) + volume spike + 1d uptrend
            if (close[i] > kama[i] and 
                rsi[i] > 50 and
                chop[i] < 61.8 and
                vol_spike[i] and
                trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below KAMA + RSI < 50 + Chop < 61.8 (trending) + volume spike + 1d downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] < 50 and
                  chop[i] < 61.8 and
                  vol_spike[i] and
                  trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below KAMA OR RSI < 40 OR Chop > 61.8 (ranging) OR 1d trend turns down
            if close[i] < kama[i] or rsi[i] < 40 or chop[i] > 61.8 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above KAMA OR RSI > 60 OR Chop > 61.8 (ranging) OR 1d trend turns up
            if close[i] > kama[i] or rsi[i] > 60 or chop[i] > 61.8 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals