#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volatility filter and 1w EMA50 trend filter
# Long when: price breaks above Donchian upper channel (20-period high), ATR(14) > 1.5 * ATR(50) (vol expansion), and close > 1w EMA50
# Short when: price breaks below Donchian lower channel (20-period low), ATR(14) > 1.5 * ATR(50), and close < 1w EMA50
# Exit when price returns to the opposite Donchian channel (mean reversion)
# Uses volatility expansion to avoid false breakouts in low-vol regimes; 1w EMA for major trend alignment
# Timeframe: 4h, HTF: 1w. Target: 80-180 total trades over 4 years (20-45/year) to balance edge and fee drag.

name = "4h_Donchian20_Breakout_1wEMA50_ATR_VolExp"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values  # Fix: properly extract open prices
    
    # Calculate ATR(14) and ATR(50) for volatility expansion filter
    if len(high) >= 50:
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = high[0] - low[0]  # first bar
        tr2[0] = np.nan
        tr3[0] = np.nan
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
        vol_expansion = atr_14 > (1.5 * atr_50)
    else:
        vol_expansion = np.zeros(n, dtype=bool)
    
    # Calculate Donchian channels (20-period) on 4h
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_expansion[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volatility expansion, and above 1w EMA50
            if (close[i] > donchian_high[i] and 
                open_price[i] <= donchian_high[i] and  # breakout confirmation
                vol_expansion[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, volatility expansion, and below 1w EMA50
            elif (close[i] < donchian_low[i] and 
                  open_price[i] >= donchian_low[i] and  # breakdown confirmation
                  vol_expansion[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian low (mean reversion to opposite channel)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high (mean reversion to opposite channel)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals