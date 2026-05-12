#!/usr/bin/env python3
"""
1D_ADX_TREND_FOLLOWING_BULL_BEAR
Hypothesis: Use ADX(14) > 25 to identify strong trends, then enter long when price > EMA(50) in bull regime (price above weekly pivot) or short when price < EMA(50) in bear regime (price below weekly pivot). Volume spike (2.0x 20-period) confirms institutional participation. Avoids whipsaw by requiring strong trend alignment. Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe.
"""
name = "1D_ADX_TREND_FOLLOWING_BULL_BEAR"
timeframe = "1d"
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
    
    # ADX calculation (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # EMA(50) for trend direction
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume spike: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for bull/bear regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot point: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_pivot_vals = weekly_pivot.values
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_vals)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA/ADX
        if (np.isnan(adx[i]) or 
            np.isnan(ema_50[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Strong uptrend (ADX>25) + price > EMA50 + price > weekly pivot (bull regime) + volume spike
            if (adx[i] > 25 and 
                close[i] > ema_50[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Strong downtrend (ADX>25) + price < EMA50 + price < weekly pivot (bear regime) + volume spike
            elif (adx[i] > 25 and 
                  close[i] < ema_50[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend weakens (ADX<20) OR price crosses below EMA50
            if adx[i] < 20 or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakens (ADX<20) OR price crosses above EMA50
            if adx[i] < 20 or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals