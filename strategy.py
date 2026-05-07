#!/usr/bin/env python3
name = "4h_Donchian20_12hTrend_VolumeSpike"
timeframe = "4h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian channel (20) on 4h close
    donch_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # 12h trend: EMA(20) on 12h close
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume spike: 24-period average (4 days of 4h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 24)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high with 12h uptrend and volume spike
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            uptrend_12h = ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]
            
            if close[i] > donch_high[i] and vol_condition and uptrend_12h:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low with 12h downtrend and volume spike
            elif close[i] < donch_low[i] and vol_condition and not uptrend_12h:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volume drops
            if close[i] < donch_low[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or volume drops
            if close[i] > donch_high[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(20) trend filter and volume confirmation
# - Donchian breakout captures momentum in trending markets
# - 12h EMA trend filter ensures alignment with higher timeframe direction
# - Volume spike (2x average) confirms institutional participation and reduces false breakouts
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price reverses to opposite Donchian band or volume weakens
# - Position size 0.25 targets ~30-50 trades/year, avoiding excessive fee drag
# - Combines price channel breakout with trend and volume for robust performance across regimes