#!/usr/bin/env python3
"""
Hypothesis: 12h KAMA + RSI + Chop regime filter for trend following in bull markets
and mean reversion in bear markets. Uses weekly trend filter to determine regime.
In bull markets (price > weekly EMA50): follow KAMA direction with RSI filter.
In bear markets (price < weekly EMA50): mean revert at RSI extremes.
Volume confirmation required for entries. Low trade frequency expected due to
regime-specific logic and multiple confirmations.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_kama_rsi_chop_regime_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)  # already shifted
    
    # === KAMA CALCULATION ===
    direction = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros_like(close)
    for i in range(1, len(close)):
        if volatility[i] != 0:
            er[i] = direction[i] / volatility[i]
        else:
            er[i] = 1.0
    er_ma = pd.Series(er).rolling(window=10, min_periods=10).mean().values
    sc = (er_ma * 0.6 + 0.06) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI CALCULATION ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === VOLUME CONFIRMATION ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Determine market regime from weekly trend
        bull_market = close[i] > weekly_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if bull_market:
                # In bull: exit when KAMA turns down OR RSI overbought
                if kama[i] < kama[i-1] or rsi[i] > 70:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # In bear: exit when RSI returns to neutral
                if rsi[i] >= 40 and rsi[i] <= 60:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if bull_market:
                # In bull: exit when KAMA turns up OR RSI oversold
                if kama[i] > kama[i-1] or rsi[i] < 30:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # In bear: exit when RSI returns to neutral
                if rsi[i] >= 40 and rsi[i] <= 60:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            if bull_market:
                # Bull market: trend following
                # Enter long when KAMA up AND RSI not overbought
                if kama[i] > kama[i-1] and rsi[i] < 70:
                    position = 1
                    signals[i] = 0.25
                # Enter short when KAMA down AND RSI not oversold
                elif kama[i] < kama[i-1] and rsi[i] > 30:
                    position = -1
                    signals[i] = -0.25
            else:
                # Bear market: mean reversion
                # Enter long when RSI oversold
                if rsi[i] < 30:
                    position = 1
                    signals[i] = 0.25
                # Enter short when RSI overbought
                elif rsi[i] > 70:
                    position = -1
                    signals[i] = -0.25
    
    return signals