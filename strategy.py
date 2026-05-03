#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves, especially effective in crypto trends.
# 1d EMA34 ensures alignment with the daily trend to avoid counter-trend trades.
# Volume confirmation (>1.5x 20-period EMA) filters false breakouts.
# Designed for low trade frequency (target: 12-37/year) on 12h timeframe to minimize fee drag.
# Works in both bull and bear markets by trading with the higher timeframe trend.

name = "12h_Donchian20_1dEMA34_VolumeConfirm"
timeframe = "12h"
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
    
    # Get 1d data for Donchian, EMA, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period) using PREVIOUS day's data to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Donchian upper = max(high, lookback=20), lower = min(low, lookback=20)
    donch_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume confirmation (volume > 1.5 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = df_1d['volume'].values > (1.5 * vol_ema_20)
    
    # Align 1d indicators to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high in uptrend with volume confirmation
            if high[i] > donch_high_aligned[i] and is_uptrend and volume_confirm_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low in downtrend with volume confirmation
            elif low[i] < donch_low_aligned[i] and is_downtrend and volume_confirm_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian low (reversal)
            if low[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian high (reversal)
            if high[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals