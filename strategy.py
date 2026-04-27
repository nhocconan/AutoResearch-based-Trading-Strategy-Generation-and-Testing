#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter and volume confirmation
# Donchian breakouts capture momentum; 12h EMA filter ensures alignment with higher timeframe trend
# Volume spike confirms institutional interest. Works in bull/bear by filtering breakout direction with 12h EMA.
# Target: 50-150 total trades over 4 years (~12-37/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian channels (20-period) on 12h timeframe
    donchian_high = np.full(len(df_12h), np.nan)
    donchian_low = np.full(len(df_12h), np.nan)
    
    for i in range(19, len(df_12h)):
        donchian_high[i] = np.max(high_12h[i-19:i+1])
        donchian_low[i] = np.min(low_12h[i-19:i+1])
    
    # Align Donchian levels to 4h timeframe (wait for 12h close)
    dh_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    dl_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 12h EMA trend filter (50-period)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 1.5 x 20-period average (approx 10 hours of 4h bars)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h data (19 bars), EMA (50), volume MA (20)
    start_idx = max(19, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dh_12h_aligned[i]) or np.isnan(dl_12h_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 12h EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and bullish trend
            if price > dh_12h_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: break below Donchian low with volume and bearish trend
            elif price < dl_12h_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low (mean reversion) or trend turns bearish
            if price <= dl_12h_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian high (mean reversion) or trend turns bullish
            if price >= dh_12h_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0