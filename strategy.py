#!/usr/bin/env python3
"""
1h Volume + 4h Trend + 1d Momentum
Hypothesis: Combine volume surge on 1h with 4h trend alignment and 1d momentum filter.
Long when: 1h volume > 2x average, price > 4h EMA50, and 1d RSI > 50.
Short when: 1h volume > 2x average, price < 4h EMA50, and 1d RSI < 50.
Uses volume surge for entry timing, 4h EMA for trend, 1d RSI for regime filter.
Target: 60-150 total trades over 4 years (15-37/year) with tight volume + trend conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14374_1h_volume_4h_trend_1d_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for EMA trend (once before loop)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data for RSI momentum filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d RSI(14) for momentum
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > (2.0 * vol_ma)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 14, 20) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: volume surge gone OR trend broken OR RSI flip OR stoploss
            if (not vol_surge[i] or close[i] <= ema_4h_aligned[i] or
                rsi_1d_aligned[i] < 50 or close[i] <= entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: volume surge gone OR trend broken OR RSI flip OR stoploss
            if (not vol_surge[i] or close[i] >= ema_4h_aligned[i] or
                rsi_1d_aligned[i] > 50 or close[i] >= entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: volume surge + trend alignment + RSI filter
            long_setup = vol_surge[i] and (close[i] > ema_4h_aligned[i]) and (rsi_1d_aligned[i] > 50)
            short_setup = vol_surge[i] and (close[i] < ema_4h_aligned[i]) and (rsi_1d_aligned[i] < 50)
            
            if long_setup and session_filter[i]:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif short_setup and session_filter[i]:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals