#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1w Ichimoku Cloud for trend filter + 1d Donchian breakout + volume confirmation.
# Ichimoku provides multi-dimensional trend, support/resistance, and momentum in one system.
# Long when price breaks above weekly Donchian high with price above Kumo and volume confirmation.
# Short when price breaks below weekly Donchian low with price below Kumo and volume confirmation.
# Exit when price re-enters Kumo or opposite Donchian breakout occurs.
# Designed for low trade frequency (20-40/year) to minimize fee decay while capturing major trends.

name = "4h_IchimokuTrend_DonchianBreakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Ichimoku
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:  # Need at least 1 year of weekly data
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Ichimoku components (standard periods: 9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    high_9 = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    high_26 = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    high_52 = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Chikou Span (Lagging Span): current close plotted 26 periods back
    # For trend filter, we'll use current price vs Kumo (cloud)
    
    # Kumo (Cloud): between Senkou Span A and Senkou Span B
    # Kumo top = max(Senkou A, Senkou B), Kumo bottom = min(Senkou A, Senkou B)
    # Since Senkou spans are plotted 26 periods ahead, we need to shift them back
    # For current Kumo, we use Senkou A and B from 26 periods ago
    # But for simplicity in filtering, we'll use current Senkou values with proper alignment
    
    # Align Ichimoku components to 4h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_b)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian(20) channels from daily data
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: 4h volume > 1.3x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ema[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Kumo (cloud) boundaries
        kumotop = max(senkou_a_aligned[i], senkou_b_aligned[i])
        kumobottom = min(senkou_a_aligned[i], senkou_b_aligned[i])
        
        if position == 0:
            # Long entry: price breaks above weekly Donchian high, above Kumo, with volume
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > kumotop and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly Donchian low, below Kumo, with volume
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < kumobottom and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters Kumo or breaks below weekly Donchian low
            if close[i] < kumotop or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Kumo or breaks above weekly Donchian high
            if close[i] > kumobottom or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals