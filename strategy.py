#!/usr/bin/env python3
"""
6h_WilliamsVixFix_MeanReversion
Hypothesis: The Williams VixFix indicator identifies periods of extreme fear/greed on 6h charts.
When VixFix > 0.8 (extreme fear) and price is below 6h EMA50, enter long.
When VixFix < 0.2 (extreme greed/complacency) and price is above 6h EMA50, enter short.
Uses 1d timeframe for EMA50 trend filter to avoid counter-trend trades.
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag.
Works in bull/bear via mean reversion from extremes + trend alignment filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (daily for EMA50 trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Williams VixFix (6h) ===
    # VixFix = ((Highest High in 22 periods - Low) / Highest High in 22 periods) * 100
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=22, min_periods=22).max().values
    vixfix = ((highest_high - low) / highest_high) * 100
    # Normalize to 0-1 range (typical VixFix ranges 0-100, extremes >80)
    vixfix_norm = vixfix / 100.0
    
    # === Daily EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === ATR (10-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(vixfix_norm[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long conditions: extreme fear (VixFix > 0.8) + price below daily EMA50 (uptrend filter)
            long_condition = (vixfix_norm[i] > 0.8) and (price < ema_50_1d_aligned[i])
            # Short conditions: extreme greed/complacency (VixFix < 0.2) + price above daily EMA50 (downtrend filter)
            short_condition = (vixfix_norm[i] < 0.2) and (price > ema_50_1d_aligned[i])
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit when fear subsides (VixFix < 0.5) or price reaches EMA50
            elif (vixfix_norm[i] < 0.5) or (price >= ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit when greed subsides (VixFix > 0.5) or price reaches EMA50
            elif (vixfix_norm[i] > 0.5) or (price <= ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsVixFix_MeanReversion"
timeframe = "6h"
leverage = 1.0