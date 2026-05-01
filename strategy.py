#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
# Uses Donchian channel breakouts for entry, 1w EMA34 for trend direction, and volume spike (>1.5x 20-bar MA) for confirmation.
# Designed for 1d timeframe to achieve 30-100 total trades over 4 years (7-25/year) with discrete sizing (0.25).
# Works in both bull and bear markets via volatility-based breakouts and tight entry conditions.

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    # Multi-timeframe: 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_1w_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_34_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 20, 34) + 1  # 35
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i]) or np.isnan(ema_1w_34_aligned[i]):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Donchian breakout conditions
        breakout_up = curr_high > highest_high[i-1]  # Break above previous period's high
        breakdown_down = curr_low < lowest_low[i-1]  # Break below previous period's low
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1w EMA34 slope (using previous bar's value)
        trend_up = ema_1w_34_aligned[i] > ema_1w_34_aligned[i-1]
        trend_down = ema_1w_34_aligned[i] < ema_1w_34_aligned[i-1]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume confirmation AND 1w EMA34 trending up
            if breakout_up and vol_confirm and trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakdown down AND volume confirmation AND 1w EMA34 trending down
            elif breakdown_down and vol_confirm and trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakdown (break below lowest low of lookback period)
            if curr_low < lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout (break above highest high of lookback period)
            if curr_high > highest_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals