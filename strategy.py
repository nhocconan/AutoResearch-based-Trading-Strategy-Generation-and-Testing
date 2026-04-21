#!/usr/bin/env python3
"""
1h_RSI2_4hTrend_MeanReversion_v1
Hypothesis: On 1h timeframe, use 4h EMA50 trend filter and RSI(2) extreme readings for mean reversion entries.
In bull regime (price > 4h EMA50): long when RSI(2) < 10 (oversold pullback).
In bear regime (price < 4h EMA50): short when RSI(2) > 90 (overbought bounce).
Add 08-20 UTC session filter to avoid low-liquidity hours. Use discrete sizing 0.20.
Target: 60-120 total trades over 4 years (15-30/year) by combining tight RSI extremes with HTF trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (4h for trend regime)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h EMA50 for trend regime ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === 1h RSI(2) for mean reversion signals ===
    close = prices['close'].values
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # === Session filter: 08-20 UTC ===
    # open_time is already datetime64[ms], use DatetimeIndex for .hour
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if indicators not ready
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        rsi_val = rsi[i]
        
        # Trend regime
        is_bull = price > ema_50_4h_val
        is_bear = price < ema_50_4h_val
        
        if position == 0:
            if is_bull:
                # Bull regime: long oversold pullbacks
                if rsi_val < 10:
                    signals[i] = 0.20
                    position = 1
            else:  # bear regime
                # Bear regime: short overbought bounces
                if rsi_val > 90:
                    signals[i] = -0.20
                    position = -1
        
        elif position != 0:
            # Exit conditions: RSI mean reversion or regime change
            if position == 1:  # long
                if rsi_val > 50 or not is_bull:  # RSI mean reverted or regime turned bear
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1, short
                if rsi_val < 50 or not is_bear:  # RSI mean reverted or regime turned bull
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_RSI2_4hTrend_MeanReversion_v1"
timeframe = "1h"
leverage = 1.0