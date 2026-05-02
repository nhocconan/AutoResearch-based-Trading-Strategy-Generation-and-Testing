#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channels capture structural breaks, 1w EMA50 ensures alignment with long-term trend
# Volume confirmation filters false breakouts. Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn
# Works in bull markets (breakout above upper channel + 1w EMA50 up) and bear markets (breakout below lower channel + 1w EMA50 down)

name = "12h_Donchian20_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter (EMA50) and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 1w EMA50 calculation
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from previous 1w bar (high/low)
    # Upper channel = max(high over last 20 periods), Lower channel = min(low over last 20 periods)
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(prev_high)
    low_series = pd.Series(prev_low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (wait for 1w bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above Donchian upper channel with volume confirmation and uptrend
            if high[i] > donchian_upper_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below Donchian lower channel with volume confirmation and downtrend
            elif low[i] < donchian_lower_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower channel (reversal) OR trend changes
            if low[i] < donchian_lower_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper channel (reversal) OR trend changes
            if high[i] > donchian_upper_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals