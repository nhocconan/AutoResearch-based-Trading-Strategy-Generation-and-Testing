#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA trend filter and volume confirmation.
# The 12h timeframe balances responsiveness with low transaction costs.
# Trend filter (1d EMA34) ensures trades align with higher timeframe direction.
# Volume confirmation avoids breakouts on low liquidity.
# Designed to work in both bull (catching breakouts) and bear (short breakdowns) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 12h data for Donchian channels and volume ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h Donchian channels (20-period)
    donch_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_12h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_12h, donch_low_20)
    
    # 12h volume average (20-period)
    vol_avg20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_avg20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg20_12h)
    
    signals = np.zeros(n)
    warmup = 34
    position = 0
    
    for i in range(warmup, n):
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or
            np.isnan(vol_avg20_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        vol_12h_current = align_htf_to_ltf(prices, df_12h, volume_12h)[i]
        vol_filter = vol_12h_current > 1.5 * vol_avg20_12h_aligned[i]
        
        if position == 0:
            # Long: breakout above Donchian high + 1d uptrend + volume filter
            if close[i] > donch_high_20_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: breakdown below Donchian low + 1d downtrend + volume filter
            if close[i] < donch_low_20_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long on breakdown or trend reversal
            if close[i] < donch_low_20_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short on breakout or trend reversal
            if close[i] > donch_high_20_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0