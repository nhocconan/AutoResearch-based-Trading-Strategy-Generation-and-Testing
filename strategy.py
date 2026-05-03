#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel AND close > 1w EMA50 AND volume > 1.5x 20-period MA.
# Short when price breaks below lower Donchian channel AND close < 1w EMA50 AND volume > 1.5x 20-period MA.
# Exit when price touches the opposite Donchian band OR volume drops below average.
# Designed to capture strong trends in both bull and bear markets with tight entries to minimize fee drag.
# Target: 30-100 total trades over 4 years (7-25/year) with discrete sizing 0.25.

name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
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
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 1d
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime: current 1d volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: Price breaks above upper Donchian AND close > 1w EMA50 AND volume spike
            if close_val > highest_high[i] and close_val > ema_50_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian AND close < 1w EMA50 AND volume spike
            elif close_val < lowest_low[i] and close_val < ema_50_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price touches lower Donchian OR volume drops below average
            if close_val < lowest_low[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price touches upper Donchian OR volume drops below average
            if close_val > highest_high[i] or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals