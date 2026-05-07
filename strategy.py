#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper (20) AND 12h close > 12h EMA50 AND volume > 1.5x 20-period EMA.
# Short when price breaks below 4h Donchian lower (20) AND 12h close < 12h EMA50 AND volume > 1.5x 20-period EMA.
# Uses 12h EMA for trend filter and volume momentum to reduce false breakouts.
# Designed for 20-40 trades/year to minimize fee drag and improve generalization.
name = "4h_Donchian20_12hEMA50_Volume"
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
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # 12h EMA50
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above Donchian upper, 12h EMA50 uptrend, volume confirmation
            long_cond = (close[i] > high_max_20[i]) and (ema50_12h_aligned[i] > ema50_12h_aligned[i-1]) and vol_confirm[i]
            # Short condition: break below Donchian lower, 12h EMA50 downtrend, volume confirmation
            short_cond = (close[i] < low_min_20[i]) and (ema50_12h_aligned[i] < ema50_12h_aligned[i-1]) and vol_confirm[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian lower or 12h EMA50 turns down
            if (close[i] < low_min_20[i]) or (ema50_12h_aligned[i] < ema50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian upper or 12h EMA50 turns up
            if (close[i] > high_max_20[i]) or (ema50_12h_aligned[i] > ema50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals