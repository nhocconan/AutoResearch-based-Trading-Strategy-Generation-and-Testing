#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume spike confirmation
# Enters long when price breaks above 20-period high with 12h EMA50 uptrend and volume > 1.5x average
# Enters short when price breaks below 20-period low with 12h EMA50 downtrend and volume > 1.5x average
# Uses discrete position sizing of 0.25 to manage risk and reduce fee churn
# Designed to capture strong trending moves with confirmation, effective in both bull and bear markets
# Target: 20-40 trades/year per symbol

name = "4h_Donchian_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4x Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:  # need 20 periods for Donchian
            highest_high[i] = np.max(high[i-19:i+1])
            lowest_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume average for volume spike filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with trend and volume confirmation
            if close[i] > highest_high[i] and ema_50_12h_aligned[i] > close_12h[-1] if len(close_12h) > 0 else False and vol_spike:
                # Additional check: ensure we're using valid 12h EMA value
                if not np.isnan(ema_50_12h_aligned[i]):
                    signals[i] = 0.25
                    position = 1
            elif close[i] < lowest_low[i] and ema_50_12h_aligned[i] < close_12h[-1] if len(close_12h) > 0 else False and vol_spike:
                # Additional check: ensure we're using valid 12h EMA value
                if not np.isnan(ema_50_12h_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price returns to middle of Donchian channel or trend weakens
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < donchian_mid or ema_50_12h_aligned[i] < close_12h[-1] if len(close_12h) > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle of Donchian channel or trend weakens
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > donchian_mid or ema_50_12h_aligned[i] > close_12h[-1] if len(close_12h) > 0 else False:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals