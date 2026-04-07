#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with 12h trend filter and volume confirmation
# Uses Donchian(20) breakout for entry, 12h EMA trend filter for direction,
# and volume spike confirmation to avoid false breakouts. Designed for low
# trade frequency (target: 12-37 trades/year) to minimize fee drift.
# Works in bull markets via breakout follow-through and in bear markets via
# breakdown continuation with trend alignment.

name = "12h_donchian20_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h EMA for trend filter
    ema_fast = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_slow = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume spike detector (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: fast EMA above/below slow EMA
        uptrend = ema_fast[i] > ema_slow[i]
        downtrend = ema_fast[i] < ema_slow[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Long: price breaks above Donchian high + uptrend + volume spike
        if close[i] > donchian_high_aligned[i] and uptrend and vol_spike:
            signals[i] = 0.25
        # Short: price breaks below Donchian low + downtrend + volume spike
        elif close[i] < donchian_low_aligned[i] and downtrend and vol_spike:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals