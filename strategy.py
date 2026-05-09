#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day trend filter and 1-week volatility filter.
# Uses 1-day EMA34 for trend direction and 1-week ATR for volatility filtering.
# Trades only in direction of daily trend when weekly volatility is elevated (ATR > 20-period average).
# Exit when trend reverses or volatility drops below average.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_EMA34_Trend_ATR_Volatility_Filter"
timeframe = "12h"
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
    
    # Get 1-day data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1-day EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1-week data for ATR volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1-week ATR(14)
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1w = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w, additional_delay_bars=0)
    
    # Calculate 20-period average of 1-week ATR for volatility filter
    atr_ma_20 = pd.Series(atr_14_1w_aligned).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr_14_1w_aligned > atr_ma_20  # Trade when volatility is above average
    
    # Trend direction: price above/below 1-day EMA34
    trend_up = close > ema_34_1d_aligned
    trend_down = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or
            np.isnan(atr_ma_20[i]) or np.isnan(trend_up[i]) or np.isnan(trend_down[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above 1-day EMA34 + volatility above average
            if trend_up[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below 1-day EMA34 + volatility above average
            elif trend_down[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reverses or volatility drops below average
            if not trend_up[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reverses or volatility drops below average
            if not trend_down[i] or not volatility_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals