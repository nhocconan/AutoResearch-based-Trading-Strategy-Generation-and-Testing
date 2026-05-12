#!/usr/bin/env python3
"""
1H_KAMA_TREND_WITH_VOLUME_CONFIRMATION
Hypothesis: KAMA adapts to market noise - in trending markets it follows price closely,
in ranging markets it stays flat. Combined with volume confirmation (>1.5x 20-bar average)
and session filter (08-20 UTC) to reduce noise. Uses 4h trend filter (EMA21) to avoid
counter-trend trades. Designed for 15-30 trades/year on 1h to minimize fee drag while
capturing sustained moves in both bull and bear markets.
"""
name = "1H_KAMA_TREND_WITH_VOLUME_CONFIRMATION"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # KAMA calculation
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(10))  # 10-period net change
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = direction / volatility.replace(0, np.nan)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 4h EMA21 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    ema21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):  # Start after warmup for KAMA
        if (np.isnan(kama[i]) or 
            np.isnan(ema21_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Only trade during session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above KAMA + volume spike + above 4h EMA21 (uptrend)
            if (close[i] > kama[i] and 
                volume_spike[i] and 
                close[i] > ema21_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price below KAMA + volume spike + below 4h EMA21 (downtrend)
            elif (close[i] < kama[i] and 
                  volume_spike[i] and 
                  close[i] < ema21_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below KAMA OR trend reversal
            if close[i] < kama[i] or close[i] < ema21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above KAMA OR trend reversal
            if close[i] > kama[i] or close[i] > ema21_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals