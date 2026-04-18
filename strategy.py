#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets, while EMA50 filter ensures
# alignment with weekly trend to avoid counter-trend trades. Volume confirmation
# adds conviction to breakouts. Designed for low trade frequency (7-25/year) on 1d
# timeframe to minimize fee drag. Works in bull markets (breakouts above upper band
# in uptrend) and bear markets (breakouts below lower band in downtrend).
name = "1d_Donchian20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian channel AND uptrend AND volume confirmed
            if close[i] > upper_channel[i] and uptrend and volume_confirmed[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel AND downtrend AND volume confirmed
            elif close[i] < lower_channel[i] and downtrend and volume_confirmed[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian channel OR trend reverses
            if close[i] < lower_channel[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian channel OR trend reverses
            if close[i] > upper_channel[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals