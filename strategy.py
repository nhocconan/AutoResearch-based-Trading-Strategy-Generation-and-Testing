#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR-based Volatility Contraction Breakout with 12h Trend Filter
# - Identify low volatility periods using ATR contraction (ATR(7) < 0.5 * ATR(30))
# - Breakout triggers when price moves beyond Donchian(20) channels
# - 12h EMA50 filter ensures breakouts align with medium-term trend
# - Designed to capture explosive moves after consolidation in both bull and bear markets
# - Target: 15-35 trades per year per symbol (60-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h timeframe
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR components on 6h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR(7) and ATR(30)
    atr_7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after ATR(30) and Donchian warmup
        # Skip if NaN in indicators
        if (np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or
            np.isnan(ema_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility contraction condition: ATR(7) < 0.5 * ATR(30)
        vol_contract = atr_7[i] < 0.5 * atr_30[i]
        
        price = close[i]
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        ema_trend = ema_12h_aligned[i]
        
        if position == 0:
            # Long entry: volatility contraction + breakout above upper channel + uptrend
            if vol_contract and price > upper_channel and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short entry: volatility contraction + breakdown below lower channel + downtrend
            elif vol_contract and price < lower_channel and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA or volatility expands significantly
            if price < ema_trend or atr_7[i] > 1.5 * atr_30[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA or volatility expands significantly
            if price > ema_trend or atr_7[i] > 1.5 * atr_30[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolContraction_Breakout_12hEMAFilter"
timeframe = "6h"
leverage = 1.0