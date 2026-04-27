#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d trend filter.
# Long when price breaks above Donchian(20) high with volume > 1.5x average and 1d close > EMA34.
# Short when price breaks below Donchian(20) low with volume > 1.5x average and 1d close < EMA34.
# Exit when price returns to Donchian middle or trend reverses.
# Designed for ~25-35 trades/year with strict entry conditions to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Get 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.5x 30-period average
    vol_ma_30 = np.full(n, np.nan)
    for i in range(29, n):
        vol_ma_30[i] = np.mean(volume[i-29:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period Donchian and 30-period volume MA
    start_idx = max(20, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_30[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filters from 1d EMA34
        bullish_trend = price > ema34_aligned[i]
        bearish_trend = price < ema34_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and bullish trend
            if price > donchian_high[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume and bearish trend
            elif price < donchian_low[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns below Donchian mid or trend turns bearish
            if price < donchian_mid[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns above Donchian mid or trend turns bullish
            if price > donchian_mid[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0