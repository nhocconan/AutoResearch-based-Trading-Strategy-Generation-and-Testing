#!/usr/bin/env python3
"""
6h_Three_Way_Confluence_v1
Hypothesis: Use 6h price action with 1d confluence filters - trade only when:
1. Price breaks 6h Donchian channel (20-period) in direction of 1d trend
2. Confirmed by 1d volume spike (>2x 20-period average)
3. Filtered by 1d ADX > 25 (strong trend) to avoid chop
Works in bull/bear by following 1d trend direction. Target: 25-40 trades/year on 6h.
"""

name = "6h_Three_Way_Confluence_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D Data for Confluence Filters ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Trend: EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d Volume spike: current volume > 2x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    
    # 1d ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(data, period):
            smoothed = np.zeros_like(data)
            smoothed[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
            return smoothed
        
        if len(tr) < period:
            return np.full_like(high, 50.0)  # neutral when insufficient data
            
        atr = wilder_smooth(tr, period)
        plus_di = 100 * wilder_smooth(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * wilder_smooth(minus_dm, period) / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    strong_trend_1d = adx_1d > 25
    
    # Align 1D indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    strong_trend_1d_aligned = align_htf_to_ltf(prices, df_1d, strong_trend_1d.astype(float))
    
    # 6h Donchian breakout (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-(period-1):i+1])
            lower[i] = np.min(low[i-(period-1):i+1])
        return upper, lower
    
    donch_upper, donch_lower = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(strong_trend_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 6h Donchian upper AND 1d uptrend AND volume spike AND strong trend
            if (close[i] > donch_upper[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike_1d_aligned[i] > 0.5 and 
                strong_trend_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6h Donchian lower AND 1d downtrend AND volume spike AND strong trend
            elif (close[i] < donch_lower[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike_1d_aligned[i] > 0.5 and 
                  strong_trend_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 6h Donchian lower OR loses 1d uptrend
            if close[i] < donch_lower[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above 6h Donchian upper OR loses 1d downtrend
            if close[i] > donch_upper[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals