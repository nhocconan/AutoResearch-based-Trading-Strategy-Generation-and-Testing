#!/usr/bin/env python3
"""
1h_LongShort_4h_1d_Trend_With_Volume_Confirmation
Hypothesis: Use 4h EMA(21) trend direction filtered by 1d EMA(50) regime, with 1h volume confirmation (volume > 1.3x 20-period average) for entry timing. Go long when 4h EMA(21) > 4h EMA(50) and 1d EMA(50) > 1d EMA(200) and volume spike; short when 4h EMA(21) < 4h EMA(50) and 1d EMA(50) < 1d EMA(200) and volume spike. Exit when trend reverses or volume dries up. Designed for 1h timeframe with strict entry conditions to limit trades to 15-37/year and avoid fee drag. Works in bull markets via 4h trend and in bear markets via 1d regime filter preventing false signals.
"""

name = "1h_LongShort_4h_1d_Trend_With_Volume_Confirmation"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA crossover signal
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA21 and EMA50
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h EMA crossover signal: 1 = bullish, -1 = bearish, 0 = neutral
    ema_cross_4h = np.where(ema21_4h > ema50_4h, 1, np.where(ema21_4h < ema50_4h, -1, 0))
    
    # Align 4h EMA crossover to 1h timeframe
    ema_cross_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_cross_4h)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 and EMA200 for regime filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1d regime: 1 = bullish (EMA50 > EMA200), -1 = bearish (EMA50 < EMA200), 0 = neutral
    regime_1d = np.where(ema50_1d > ema200_1d, 1, np.where(ema50_1d < ema200_1d, -1, 0))
    
    # Align 1d regime to 1h timeframe
    regime_1d_aligned = align_htf_to_ltf(prices, df_1d, regime_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_cross_4h_aligned[i]) or np.isnan(regime_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.3x 20-period average
        vol_spike = volume[i] > 1.3 * vol_ma_20[i]
        
        if position == 0:
            # LONG: 4h bullish crossover + 1d bullish regime + volume spike
            if ema_cross_4h_aligned[i] == 1 and regime_1d_aligned[i] == 1 and vol_spike:
                signals[i] = 0.20
                position = 1
            # SHORT: 4h bearish crossover + 1d bearish regime + volume spike
            elif ema_cross_4h_aligned[i] == -1 and regime_1d_aligned[i] == -1 and vol_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h bearish crossover or 1d bearish regime
            if ema_cross_4h_aligned[i] == -1 or regime_1d_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h bullish crossover or 1d bullish regime
            if ema_cross_4h_aligned[i] == 1 or regime_1d_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals