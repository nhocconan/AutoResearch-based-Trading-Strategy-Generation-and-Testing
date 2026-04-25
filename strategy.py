#!/usr/bin/env python3
"""
1d Donchian(20) breakout + 1w EMA34 trend + volume confirmation
Hypothesis: Daily Donchian channel breakouts capture strong momentum, filtered by weekly EMA trend to avoid counter-trend trades and volume confirmation to ensure conviction. Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year), minimizing fee drag. Works in bull markets via long breakouts and bear markets via short breakouts, with trend filter preventing entries against the weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20)
    low_roll = pd.Series(low).rolling(window=20, min_periods=20)
    donchian_high = high_roll.max().shift(1).values  # breakout above previous period high
    donchian_low = low_roll.min().shift(1).values   # breakdown below previous period low
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20) + volume MA (20) + weekly EMA warmup
    start_idx = max(20, 20, 34)  # Donchian, volume MA, weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to weekly EMA34
        bullish_bias = curr_close > ema_1w_aligned[i]
        bearish_bias = curr_close < ema_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian high AND bullish bias AND volume spike
            long_entry = (curr_high > donchian_high[i]) and bullish_bias and vol_spike
            # Short: price breaks below Donchian low AND bearish bias AND volume spike
            short_entry = (curr_low < donchian_low[i]) and bearish_bias and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian low (mean reversion) OR loss of bullish bias
            if (curr_low < donchian_low[i]) or (curr_close < ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian high (mean reversion) OR loss of bearish bias
            if (curr_high > donchian_high[i]) or (curr_close > ema_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0