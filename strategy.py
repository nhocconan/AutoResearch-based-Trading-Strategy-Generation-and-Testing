#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume confirmation and 1w trend filter.
# Long when price breaks above Donchian(20) high with 1d volume > 1.5x average and 1w close > EMA50.
# Short when price breaks below Donchian(20) low with 1d volume > 1.5x average and 1w close < EMA50.
# Exit when price returns to Donchian(20) midline or volume drops below average.
# Designed for ~20-30 trades/year with strong trend and volume filters to avoid whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 4h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate daily volume average (20-day)
    vol_ma_20 = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align daily volume average to 4h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period Donchian and 20-day volume MA
    start_idx = max(19, 19)  # Need enough data for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg_1d = vol_ma_20_aligned[i]
        
        # Volume filter: current volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_avg_1d
        
        # Trend filter from weekly EMA50
        bullish_trend = price > ema50_1w_aligned[i]
        bearish_trend = price < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and bullish trend
            if price > donchian_high[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: break below Donchian low with volume and bearish trend
            elif price < donchian_low[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midline or volume drops
            if price < donchian_mid[i] or vol_now <= vol_avg_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to Donchian midline or volume drops
            if price > donchian_mid[i] or vol_now <= vol_avg_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_1wTrend"
timeframe = "4h"
leverage = 1.0