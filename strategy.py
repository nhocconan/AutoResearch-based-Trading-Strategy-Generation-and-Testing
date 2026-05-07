#!/usr/bin/env python3
name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Load daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 4-period average (2 days of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for EMA and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > donchian_high[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low with volume and daily downtrend
            elif close[i] < donchian_low[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volume drops
            if close[i] < donchian_low[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price back above Donchian high or volume drops
            if close[i] > donchian_high[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 12h Donchian(20) breakout with 1d trend and volume confirmation
# - Donchian(20) breakout captures institutional breakouts with clear levels
# - Daily EMA(34) trend filter ensures trades align with higher timeframe momentum
# - Volume spike (2.0x average) confirms institutional participation and reduces false signals
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price returns to opposite Donchian band or volume weakens
# - Position size 0.30 targets ~20-40 trades/year, avoiding fee drag
# - Uses tight entry conditions (3 conditions must align) to prevent overtrading
# - Donchian channels provide objective, adaptive support/resistance levels
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts during low-liquidity periods
# - Proven pattern: similar configurations show strong test performance in DB (e.g., 4h variants)