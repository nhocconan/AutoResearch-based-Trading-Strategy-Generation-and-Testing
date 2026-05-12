#!/usr/bin/env python3
# 4H_MULTI_TIMEFRAME_RSI_WITH_ATR_FILTER
# Hypothesis: Combines RSI momentum (14-period) with 1d trend filter (EMA50) and ATR volatility filter to capture medium-term reversals in BTC/ETH.
# Long when RSI crosses above 30 from below with ATR expansion and price above 1d EMA50.
# Short when RSI crosses below 70 from above with ATR expansion and price below 1d EMA50.
# Exit when RSI returns to opposite extreme (70 for long, 30 for short) or trend invalidates.
# Designed for 4h timeframe to balance trade frequency and signal quality, targeting 20-40 trades/year.

name = "4H_MULTI_TIMEFRAME_RSI_WITH_ATR_FILTER"
timeframe = "4h"
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
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values  # 10-period ATR EMA for expansion
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    pclose = df_1d['close'].values
    ema1d = pd.Series(pclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(rsi[i]) or np.isnan(atr[i]) or np.isnan(atr_ma[i]) or np.isnan(ema1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 1.1x ATR EMA (expansion)
        vol_expansion = atr[i] > atr_ma[i] * 1.1
        
        if position == 0:
            # LONG: RSI crosses above 30 from below with vol expansion and uptrend
            if rsi[i] > 30 and rsi[i-1] <= 30 and vol_expansion and close[i] > ema1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI crosses below 70 from above with vol expansion and downtrend
            elif rsi[i] < 70 and rsi[i-1] >= 70 and vol_expansion and close[i] < ema1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to overbought or trend breaks
            if rsi[i] >= 70 or close[i] < ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to oversold or trend breaks
            if rsi[i] <= 30 or close[i] > ema1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals