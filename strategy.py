#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with weekly EMA34 trend filter and volume spike confirmation.
Targets 7-25 trades/year by requiring: 1) price breaks 20-day Donchian channel, 2) aligned with 1w EMA34 trend,
3) volume > 1.5x 20-day average volume. Uses discrete position sizing (0.25) to minimize fee churn.
Works in bull markets via trend-following breaks and in bear markets via mean-reversion exits at opposing Donchian levels.
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
    prev_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    prev_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    avg_volume_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Align 1d levels to 1d timeframe (no alignment needed as we're already on 1d)
    upper_channel = prev_high_20
    lower_channel = prev_low_20
    vol_threshold = avg_volume_20 * 1.5
    
    # 1w data for EMA34 trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (20+1) + 1w EMA34 (34)
    start_idx = 20 + 1 + 34  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_threshold[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: price relative to 1w EMA34
        uptrend = curr_close > ema_34_1w_aligned[i]
        downtrend = curr_close < ema_34_1w_aligned[i]
        
        if position == 0:
            # Look for entry signals with trend alignment and volume confirmation
            vol_confirm = curr_volume > vol_threshold[i]
            
            # Long breakout: price breaks above upper channel with uptrend and volume
            long_breakout = (curr_close > upper_channel[i]) and uptrend and vol_confirm
            # Short breakout: price breaks below lower channel with downtrend and volume
            short_breakout = (curr_close < lower_channel[i]) and downtrend and vol_confirm
            
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
            # Long position: exit if price breaks below lower channel (mean reversion) or trend changes to downtrend
            if curr_close < lower_channel[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit if price breaks above upper channel (mean reversion) or trend changes to uptrend
            if curr_close > upper_channel[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0