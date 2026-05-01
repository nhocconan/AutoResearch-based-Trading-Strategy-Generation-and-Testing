#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses 1w EMA34 for structural trend bias (long when price > EMA34, short when price < EMA34)
# Donchian(20) levels provide robust breakout structure from daily range
# Volume confirmation > 1.8x 20-period EMA ensures institutional participation
# Designed for low trade frequency: ~7-25 trades/year per symbol with 0.28 sizing
# 1w EMA34 filter reduces false breakouts in choppy markets while capturing strong trends
# Works in both bull and bear markets by following the dominant weekly trend

name = "1d_Donchian20_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Donchian(20) levels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.8 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1w data for EMA34 (35 weeks) + Donchian needs 20 days
    start_idx = 35
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA34: long above EMA34, short below EMA34
        bullish_bias = close[i] > ema_34_1w_aligned[i]
        bearish_bias = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Donchian(20) breakout above with volume spike
                if close[i] > highest_high[i] and volume_spike[i]:
                    signals[i] = 0.28
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Donchian(20) breakdown below with volume spike
                if close[i] < lowest_low[i] and volume_spike[i]:
                    signals[i] = -0.28
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA34
        
        elif position == 1:  # Long position
            # Exit: Donchian(20) breakdown below (failure of breakout)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit: Donchian(20) breakout above (failure of breakdown)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals