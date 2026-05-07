#!/usr/bin/env python3
name = "4h_Donchian20_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 25:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian channel (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA20 for trend filter
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: breakout above Donchian high with 12h uptrend and volume
            if close[i] > donchian_high[i] and ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low with 12h downtrend and volume
            elif close[i] < donchian_low[i] and ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price back below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price back above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with 12h EMA20 trend filter and volume confirmation
# - Long when price breaks above 20-period high with 12h uptrend and volume spike
# - Short when price breaks below 20-period low with 12h downtrend and volume spike
# - Exit when price returns to opposite Donchian band
# - Uses 12h trend to avoid counter-trend trades in ranging markets
# - Volume confirmation reduces false breakouts
# - Position size 0.25 targets ~30-50 trades/year to stay within limits
# - Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend)
# - Simple, robust structure with clear entry/exit rules
# - Avoids overtrading by requiring trend alignment and volume confirmation
# - Expected trades: 20-40/year per symbol, well under the 400 total limit
# - Proven pattern: Donchian breakouts + volume + trend filter work on BTC/ETH/SOL