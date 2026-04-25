#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with 1-week EMA34 trend filter and volume spike confirmation.
Targets 7-25 trades/year by requiring: 1) price breaks 20-day Donchian channel, 2) aligned with 1w EMA34 trend,
3) volume > 1.5x 20-day average volume. Uses 1d timeframe to minimize fee drag and capture significant moves.
The volume spike filter avoids breakouts in low-participation markets, improving performance in both bull and bear markets.
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
    
    # 1d data for Donchian channels and volume average (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 20-day Donchian channels (using previous day's data to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    upper_channel = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d channels to 1d timeframe (no shift needed as we used shift(1) above)
    upper_aligned = upper_channel  # Already aligned to 1d
    lower_aligned = lower_channel  # Already aligned to 1d
    
    # 1w EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 20-day average volume for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + 1d Donchian (20) + 1w EMA34 (34) + 1d volume MA (20)
    start_idx = 34 + 20 + 1  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price relative to 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        # Volume confirmation: volume > 1.5x 20-day average
        volume_spike = curr_volume > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for entry signals with trend and volume confirmation
            # Long breakout: price breaks above upper channel with uptrend and volume spike
            long_breakout = (curr_close > upper_aligned[i]) and uptrend and volume_spike
            # Short breakout: price breaks below lower channel with downtrend and volume spike
            short_breakout = (curr_close < lower_aligned[i]) and downtrend and volume_spike
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit if price breaks below lower channel (mean reversion)
            if curr_close < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above upper channel (mean reversion)
            if curr_close > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0