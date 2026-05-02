#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag
# Donchian(20) provides clear breakout structure from recent price extremes
# 1w EMA50 determines long-term trend bias: long when price > EMA50, short when price < EMA50
# Volume spike (2x 20-period average) confirms institutional participation
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.30 (30% of capital) balances exposure and risk
# Uses 1w as HTF as specified in experiment #117281

name = "4h_Donchian20_1wEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Donchian levels (prior completed 1w bar's range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior completed 1w bar's high, low, close
    ph = df_1w['high'].shift(1).values  # prior week high
    pl = df_1w['low'].shift(1).values   # prior week low
    pc = df_1w['close'].shift(1).values # prior week close
    
    # Donchian upper/lower bands (20-period on 1w)
    donchian_upper = pd.Series(ph).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(pl).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe (wait for completed 1w bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Calculate 1w EMA50 trend (prior completed 1w bar's EMA)
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 4h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper AND price > 1w EMA50 (bullish bias) AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Donchian lower AND price < 1w EMA50 (bearish bias) AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian lower OR below 1w EMA50 (trend change)
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper OR above 1w EMA50 (trend change)
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals