#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian channels provide clear breakout levels based on recent price extremes
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend whipsaws
# Volume spike (>1.8x 20-period average) confirms institutional participation
# Discrete sizing 0.28 balances profit potential with drawdown control
# Works in bull markets via breakouts and bear markets via breakdowns with trend filter
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels: 20-period high/low
    # Using 20 * 4h = 80h ≈ 3.3 days of price action
    lookback = 20
    highest_20 = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_20 = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20)  # warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and price above/below EMA34_1d for trend alignment
            if curr_volume_spike:
                # Bullish entry: break above 20-period high with price > EMA34_1d
                if curr_close > curr_highest_20 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.28
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below 20-period low with price < EMA34_1d
                elif curr_close < curr_lowest_20 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.28
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below 20-period low OR price crosses below EMA34_1d
            if curr_close < curr_lowest_20 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Exit when price rises above 20-period high OR price crosses above EMA34_1d
            if curr_close > curr_highest_20 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals