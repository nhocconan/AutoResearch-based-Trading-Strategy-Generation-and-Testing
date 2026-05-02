#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Donchian breakouts capture institutional momentum moves. Weekly pivot (from 1w OHLC) provides
# higher timeframe directional bias to avoid counter-trend whipsaws. Volume confirmation ensures
# breakout authenticity. Designed for 6h timeframe targeting 12-37 trades/year (50-150 total).
# Works in bull markets (breakout above upper band + weekly pivot up) and bear markets
# (breakout below lower band + weekly pivot down). Uses discrete sizing (0.25) to control fees/drawdown.

name = "6h_Donchian20_1wPivot_Dir_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for pivot direction (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for pivot calculation
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3
    # Weekly bias: close > P = up-trend, close < P = down-trend
    weekly_pivot = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3
    weekly_bias_up = df_1w['close'].values > weekly_pivot  # True if weekly close above pivot
    weekly_bias_down = df_1w['close'].values < weekly_pivot  # True if weekly close below pivot
    
    # Align weekly bias to 6h timeframe (use previous week's bias to avoid look-ahead)
    weekly_bias_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_up.astype(float))
    weekly_bias_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_down.astype(float))
    
    # Donchian channels (20-period) on 6h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(weekly_bias_up_aligned[i]) or np.isnan(weekly_bias_down_aligned[i]) or
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian upper band with weekly up bias and volume
            if close[i] > highest_high[i] and weekly_bias_up_aligned[i] > 0.5 and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian lower band with weekly down bias and volume
            elif close[i] < lowest_low[i] and weekly_bias_down_aligned[i] > 0.5 and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower band (reversal)
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper band (reversal)
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals