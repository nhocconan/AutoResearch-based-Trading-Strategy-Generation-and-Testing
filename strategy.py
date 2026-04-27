#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d trend filter.
# Long when price breaks above 4h Donchian high (20) with volume > 1.5x average and 1d close > EMA50.
# Short when price breaks below 4h Donchian low (20) with volume > 1.5x average and 1d close < EMA50.
# Exit when price crosses 4h Donchian midline or volume drops below average.
# Uses 4h for direction (trend/breakout), 1h for entry timing, 1d for trend filter.
# Target: 15-30 trades/year to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donch_high_4h = rolling_max(high_4h, 20)
    donch_low_4h = rolling_min(low_4h, 20)
    donch_mid_4h = (donch_high_4h + donch_low_4h) / 2.0
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, donch_mid_4h)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 24-period average
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need 20-period Donchian and 24-period volume MA
    start_idx = max(20, 23)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Breakout conditions
        bullish_breakout = price > donch_high_aligned[i]
        bearish_breakout = price < donch_low_aligned[i]
        
        # Trend filter from 1d EMA50
        bullish_trend = price > ema50_aligned[i]
        bearish_trend = price < ema50_aligned[i]
        
        if position == 0:
            # Long: bullish breakout with volume and bullish trend
            if bullish_breakout and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: bearish breakout with volume and bearish trend
            elif bearish_breakout and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses Donchian midline or volume drops
            if price < donch_mid_aligned[i] or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses Donchian midline or volume drops
            if price > donch_mid_aligned[i] or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_DonchianBreakout_Volume_1dTrend"
timeframe = "1h"
leverage = 1.0