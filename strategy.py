#!/usr/bin/env python3
# 4h_Donchian20_1dTrend_Volume
# Hypothesis: Uses 4h Donchian channel breakouts filtered by 1-day EMA trend and volume confirmation.
# Long when price breaks above 20-period Donchian high with volume surge and daily uptrend.
# Short when price breaks below 20-period Donchian low with volume surge and daily downtrend.
# Exits when price crosses the Donchian midline (10-period average) or trend reverses.
# Designed for 4h timeframe with 1d trend filter to capture medium-term momentum while minimizing whipsaws.
# Target: 20-40 trades/year to stay within fee-efficient range.

name = "4h_Donchian20_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d EMA50 for trend filter ---
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- 4h Donchian Channel (20-period) ---
    # Calculate rolling max/min for high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0  # Midline for exit
    
    # --- Volume confirmation (2x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for 1d EMA50 (50) and 20-period Donchian/volume
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume surge and daily uptrend
            if (close[i] > donchian_high[i] and 
                volume_surge and 
                ema_50_1d_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume surge and daily downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_surge and 
                  ema_50_1d_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below Donchian midline OR daily EMA50 turns down
                if (close[i] < donchian_mid[i] or 
                    close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above Donchian midline OR daily EMA50 turns up
                if (close[i] > donchian_mid[i] or 
                    close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals