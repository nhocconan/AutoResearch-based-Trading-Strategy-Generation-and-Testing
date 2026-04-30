#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Donchian(20) provides clear structure for breakouts in both bull and bear markets
# 1d EMA34 ensures alignment with medium-term daily trend to avoid counter-trend whipsaws
# Volume spike (2.0x 96-period average) confirms institutional participation on 4h timeframe
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull markets via breakouts above upper band and bear markets via breakdowns below lower band with trend filter.

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 96-period average (96*4h = 384h = 16 days)
    vol_ma_96 = pd.Series(volume).rolling(window=96, min_periods=96).mean().values
    volume_spike = volume > (2.0 * vol_ma_96)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 96)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_96[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_high_20 = high_20[i]
        curr_low_20 = low_20[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment (price above/below EMA34)
            if curr_volume_spike:
                # Bullish entry: break above upper Donchian with price above EMA34
                if curr_close > curr_high_20 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: break below lower Donchian with price below EMA34
                elif curr_close < curr_low_20 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below lower Donchian (breakout fails) OR trend reverses (price below EMA34)
            if curr_close < curr_low_20 or curr_close < curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above upper Donchian (breakdown fails) OR trend reverses (price above EMA34)
            if curr_close > curr_high_20 or curr_close > curr_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals