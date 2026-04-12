#!/usr/bin/env python3
"""
4h_12h_KAMA_Trend_Reversal
Hypothesis: On 4h timeframe, take long when KAMA turns upward from oversold (RSI<30) in uptrend,
and short when KAMA turns downward from overbought (RSI>70) in downtrend.
Use 12h ADX>25 for trend strength filter and volume > 1.5x average for confirmation.
Exit when RSI reaches opposite extreme or trend weakens (ADX<20).
Designed to capture mean-reversion within strong trends with low trade frequency.
Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_KAMA_Trend_Reversal"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H DATA FOR TREND STRENGTH (ADX) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    atr = np.convolve(tr, np.ones(14)/14, mode='same')
    atr[:13] = np.nan
    
    plus_di = 100 * np.convolve(plus_dm, np.ones(14)/14, mode='same') / atr
    minus_di = 100 * np.convolve(minus_dm, np.ones(14)/14, mode='same') / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = np.convolve(dx, np.ones(14)/14, mode='same')
    
    adx[:27] = np.nan
    adx_12h = adx
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === KAMA CALCULATION ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) >= 10 else np.full_like(close, np.nan)
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # === RSI CALCULATION ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.convolve(gain, np.ones(14)/14, mode='same')
    avg_loss = np.convolve(loss, np.ones(14)/14, mode='same')
    avg_gain[:13] = np.nan
    avg_loss[:13] = np.nan
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === VOLUME FILTER ===
    vol_ma = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma[:19] = np.nan
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(kama_slope[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend strength filter
        strong_trend = adx_12h_aligned[i] > 25
        weak_trend = adx_12h_aligned[i] < 20
        
        # KAMA direction
        kama_up = kama_slope[i] > 0
        kama_down = kama_slope[i] < 0
        
        # RSI extremes
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Entry conditions
        long_signal = kama_up and rsi_oversold and strong_trend and vol_ratio[i] > 1.5
        short_signal = kama_down and rsi_overbought and strong_trend and vol_ratio[i] > 1.5
        
        # Exit conditions
        exit_long = (position == 1 and (rsi[i] > 70 or weak_trend))
        exit_short = (position == -1 and (rsi[i] < 30 or weak_trend))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals