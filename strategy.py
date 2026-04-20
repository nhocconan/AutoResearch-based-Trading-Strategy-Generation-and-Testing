#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian Breakout with 12h Volume Confirmation and 1d Trend Filter
# - Long when price breaks above 6h Donchian Upper (20-period) + 12h volume > 12h VWAP-volume ratio + 1d close > 1d EMA50
# - Short when price breaks below 6h Donchian Lower (20-period) + 12h volume > 12h VWAP-volume ratio + 1d close < 1d EMA50
# - Uses actual breakouts with volume confirmation to filter false signals
# - Trend filter from 1d EMA50 ensures alignment with higher timeframe trend
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h and 1d data for volume and trend filters
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h VWAP and volume ratio
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_12h = (typical_price_12h * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_12h_array = vwap_12h.values
    volume_ratio_12h = df_12h['volume'] / vwap_12h_array
    volume_ratio_12h_array = volume_ratio_12h.values
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h volume ratio and 1d EMA50 to 6h timeframe
    volume_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ratio_12h_array)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h Donchian channels (20-period)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_upper = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after Donchian and EMA warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(volume_ratio_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        vol_ratio = volume_ratio_12h_aligned[i]
        ema_trend = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + volume confirmation + uptrend
            if price > donchian_upper[i] and vol_ratio > 1.5 and close_6h[i] > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + volume confirmation + downtrend
            elif price < donchian_lower[i] and vol_ratio > 1.5 and close_6h[i] < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian lower or trend turns bearish
            if price < donchian_lower[i] or close_6h[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Donchian upper or trend turns bullish
            if price > donchian_upper[i] or close_6h[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_Volume_TrendFilter"
timeframe = "6h"
leverage = 1.0