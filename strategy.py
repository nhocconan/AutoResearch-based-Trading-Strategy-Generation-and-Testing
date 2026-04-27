#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for higher timeframe context and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA trend filter (21-period)
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Daily ATR for volatility-based stop (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Daily close price for entry reference
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly EMA (21), daily ATR (14), daily close alignment
    start_idx = max(21, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(close_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_ema = ema_21_1w_aligned[i]
        atr = atr_14_aligned[i]
        daily_close = close_1d_aligned[i]
        
        if position == 0:
            # Long: price above weekly EMA + touches/pullbacks to daily close + volatility filter
            if price > weekly_ema and abs(price - daily_close) <= 0.5 * atr:
                signals[i] = size
                position = 1
            # Short: price below weekly EMA + touches/pullbacks to daily close + volatility filter
            elif price < weekly_ema and abs(price - daily_close) <= 0.5 * atr:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA or ATR-based stop
            if price < weekly_ema or price < daily_close - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA or ATR-based stop
            if price > weekly_ema or price > daily_close + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyEMA21_DailyPullback_ATRStop"
timeframe = "1d"
leverage = 1.0