#!/usr/bin/env python3
name = "4h_Donchian20_VolumeTrend_12hEMA50"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Donchian channel (20-period) on 4h close
    donchian_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Wait for Donchian and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or \
           np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_20[i] * 1.8
            uptrend_12h = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            
            if close[i] > donchian_high[i] and vol_condition and uptrend_12h:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and 12h downtrend
            elif close[i] < donchian_low[i] and vol_condition and not uptrend_12h:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or trend reversal
            if close[i] < donchian_low[i] or ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or trend reversal
            if close[i] > donchian_high[i] or ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# - Donchian breakout captures institutional breakouts in both bull and bear markets
# - 12h EMA50 ensures trading with higher timeframe trend (avoids counter-trend whipsaws)
# - Volume confirmation (1.8x average) filters false breakouts
# - Exit on Donchian reversal or trend change to limit drawdown
# - Position size 0.25 targets ~30-50 trades/year, minimizing fee drag
# - Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)