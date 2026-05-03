#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Donchian breakout captures strong momentum moves. 1w EMA34 ensures alignment with weekly trend.
# Volume spike confirms institutional participation. Designed for low trade frequency (target: 7-25/year)
# to minimize fee drag on 1d timeframe. Works in bull and bear markets by trading with weekly trend.

name = "1d_Donchian20_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for Donchian, EMA, and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    # Use shifted data to avoid look-ahead: based on previous 20 weekly candles
    shifted_high = df_1w['high'].shift(1).values
    shifted_low = df_1w['low'].shift(1).values
    
    # Calculate rolling max/min for Donchian
    high_series = pd.Series(shifted_high)
    low_series = pd.Series(shifted_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1w volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1w['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1w indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high in uptrend with volume spike
            # OR strong breakout above Donchian high regardless of trend
            if ((high[i] > donchian_high_aligned[i] and is_uptrend and volume_spike_aligned[i]) or
                (high[i] > donchian_high_aligned[i] * 1.01)):  # 1% buffer for strong breakout
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low in downtrend with volume spike
            # OR strong breakout below Donchian low regardless of trend
            elif ((low[i] < donchian_low_aligned[i] and is_downtrend and volume_spike_aligned[i]) or
                  (low[i] < donchian_low_aligned[i] * 0.99)):  # 1% buffer for strong breakout
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian low (reversal) or hits opposite band (profit target)
            if low[i] < donchian_low_aligned[i] or high[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian high (reversal) or hits opposite band (profit target)
            if high[i] > donchian_high_aligned[i] or low[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals