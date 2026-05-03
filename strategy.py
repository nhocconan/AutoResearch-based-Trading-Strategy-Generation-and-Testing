#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channels provide clear structure-based breakout levels from prior 20-day high/low.
# Breakout above 20-day high or below 20-day low with 1w trend alignment captures major momentum moves.
# Volume confirmation filters false breakouts. Designed for 15-25 trades/year on 1d to minimize fee drag.
# Works in bull markets via buying breakouts in uptrends and bear markets via selling breakdowns in downtrends.
# Uses discrete position sizing (0.0, ±0.25) to control turnover and fees.

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Need at least 20 bars for Donchian calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian(20) using prior 20 days' high/low
        # Donchian requires prior 20 periods, so we look back 20 completed daily bars
        if i >= 21:  # Need at least 21 bars to have 20 prior bars + current
            # Get prior 20 completed daily highs and lows (excluding current forming bar)
            prior_highs = high[i-20:i]   # 20 periods before current bar
            prior_lows = low[i-20:i]
            
            donchian_high = np.max(prior_highs)
            donchian_low = np.min(prior_lows)
            
            # Avoid division by zero or invalid range
            if donchian_high <= donchian_low:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
            
            # Volume confirmation: 20-period EMA on 1d
            if i >= 19:
                vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
            else:
                vol_ema_20 = volume[i]
            volume_spike = volume[i] > (1.5 * vol_ema_20)
            
            # Donchian breakout conditions
            breakout_up = close[i] > donchian_high
            breakout_down = close[i] < donchian_low
            
            if position == 0:
                # Long: bullish breakout above 20-day high in 1w uptrend with volume spike
                if breakout_up and ema_50_1w_aligned[i] < close[i] and volume_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: bearish breakdown below 20-day low in 1w downtrend with volume spike
                elif breakout_down and ema_50_1w_aligned[i] > close[i] and volume_spike:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: price returns to 20-day high or loses 1w uptrend
                if close[i] < donchian_high or ema_50_1w_aligned[i] >= close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to 20-day low or loses 1w downtrend
                if close[i] > donchian_low or ema_50_1w_aligned[i] <= close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Not enough data for Donchian calculation yet
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals