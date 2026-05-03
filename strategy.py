#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel AND price > 1w EMA50 AND volume > 1.5x 20-period MA.
# Short when price breaks below lower Donchian channel AND price < 1w EMA50 AND volume > 1.5x 20-period MA.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year).
# Works in bull via long signals and bear via short signals when aligned with weekly trend.
# 12h timeframe reduces noise and overtrading vs lower timeframes.

name = "12h_Donchian20_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) on 12h: upper = max(high, 20), lower = min(low, 20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 12h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: Price breaks above upper Donchian AND above 1w EMA50 AND volume spike
            if close_val > highest_high[i] and close_val > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian AND below 1w EMA50 AND volume spike
            elif close_val < lowest_low[i] and close_val < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below lower Donchian OR trend reverses (below EMA50) OR volume drops
            if close_val < lowest_low[i] or close_val < ema_trend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above upper Donchian OR trend reverses (above EMA50) OR volume drops
            if close_val > highest_high[i] or close_val > ema_trend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals