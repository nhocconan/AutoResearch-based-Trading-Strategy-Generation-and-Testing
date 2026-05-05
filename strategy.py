#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1w EMA50 trend filter
# Long when: price breaks above Donchian upper (20-period high), volume > 2.0x 48-period average, and close > 1w EMA50
# Short when: price breaks below Donchian lower (20-period low), volume > 2.0x 48-period average, and close < 1w EMA50
# Exit when price returns to Donchian midpoint (mean reversion)
# Uses proven Donchian breakout structure with volume confirmation and weekly trend filter for BTC/ETH resilience in bull/bear markets
# Timeframe: 4h, HTF: 1d/1w. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Donchian20_Breakout_1wEMA50_1dVolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 4h using 48-period MA (equivalent to 1d lookback)
    if len(volume) >= 48:
        vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
        volume_filter = volume > (2.0 * vol_ma_48)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 1d data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels from previous 1d bar (20-period high/low)
    if len(high_1d) >= 20:
        # Use rolling window on 1d data for Donchian
        high_series = pd.Series(high_1d)
        low_series = pd.Series(low_1d)
        donchian_high = high_series.rolling(window=20, min_periods=20).max().values
        donchian_low = low_series.rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_high + donchian_low) / 2.0
    else:
        donchian_high = np.full(len(high_1d), np.nan)
        donchian_low = np.full(len(low_1d), np.nan)
        donchian_mid = np.full(len(high_1d), np.nan)
    
    # Align Donchian channels and 1w EMA to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)  # already computed above
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, volume filter, and above 1w EMA50
            if (close[i] > donchian_high_aligned[i] and 
                open_price[i] <= donchian_high_aligned[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low, volume filter, and below 1w EMA50
            elif (close[i] < donchian_low_aligned[i] and 
                  open_price[i] >= donchian_low_aligned[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian midpoint (mean reversion)
            if close[i] < donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian midpoint (mean reversion)
            if close[i] > donchian_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals