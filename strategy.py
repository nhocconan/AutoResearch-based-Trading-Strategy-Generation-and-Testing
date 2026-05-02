#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1d timeframe to target 30-100 trades over 4 years (7-25/year) to minimize fee drag
# Donchian channels from prior 20 days provide institutional support/resistance levels
# 1w EMA50 ensures alignment with weekly trend for higher probability trades
# Volume spike (2x 20-period average) confirms institutional participation
# Works in bull markets via breakouts and bear markets via fade of false breakouts
# Discrete position sizing: 0.30 (30% of capital) balances exposure and risk

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w Donchian levels (prior completed 1w bar's range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 weeks for prior week calculation
        return np.zeros(n)
    
    # Prior completed 1w bar's high/low for Donchian calculation
    prior_high = df_1w['high'].shift(1).values
    prior_low = df_1w['low'].shift(1).values
    
    # Calculate 1w Donchian levels: upper = prior 20-period high, lower = prior 20-period low
    # Since we're using weekly data, we need to look back 20 weeks for the channel
    upper_channel = pd.Series(prior_high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(prior_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].shift(1)).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Align HTF indicators to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper channel AND price > 1w EMA50 (bullish trend) AND volume spike
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below lower channel AND price < 1w EMA50 (bearish trend) AND volume spike
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below lower channel OR below 1w EMA50 (trend change)
            if close[i] < lower_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price rises above upper channel OR above 1w EMA50 (trend change)
            if close[i] > upper_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals