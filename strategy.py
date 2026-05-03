#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Uses Tenkan-sen/Kijun-sen cross above/below cloud from 1d timeframe for signal direction.
# Entry only when price breaks 6h Donchian(20) in direction of 1d Ichimoku trend.
# Volume spike confirms institutional participation. Discrete sizing 0.25.
# Ichimoku cloud acts as dynamic support/resistance, effective in both bull and bear regimes.
# Target: 50-150 total trades over 4 years (12-37/year).
# Focus on BTC/ETH as primary targets.

name = "6h_Ichimoku_Cloud_1dTrend_DonchianBreakout_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Ichimoku cloud calculation (using prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 52:  # Need sufficient data for Ichimoku (26*2)
        return np.zeros(n)
    
    # Calculate Ichimoku components on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # The cloud is between Senkou Span A and Senkou Span B
    # We need to shift these forward by 26 periods to align with Ichimoku definition
    # But for signal generation, we use the current cloud (which is plotted 26 periods ahead)
    # So we actually need the values that were calculated 26 periods ago
    senkou_a_aligned = np.roll(senkou_a, 26)
    senkou_b_aligned = np.roll(senkou_b, 26)
    # Set first 26 values to NaN since they don't have valid cloud data
    senkou_a_aligned[:26] = np.nan
    senkou_b_aligned[:26] = np.nan
    
    # Align Ichimoku components to 6h timeframe
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1d, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1d, kijun_sen)
    senkou_a_aligned_6h = align_htf_to_ltf(prices, df_1d, senkou_a_aligned)
    senkou_b_aligned_6h = align_htf_to_ltf(prices, df_1d, senkou_b_aligned)
    
    # Determine 1d Ichimoku trend: price above cloud = bullish, below cloud = bearish
    # We'll use the close price from 1d to determine trend
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_aligned_6h, senkou_b_aligned_6h)
    cloud_bottom = np.minimum(senkou_a_aligned_6h, senkou_b_aligned_6h)
    # Trend: 1 = bullish (price above cloud), -1 = bearish (price below cloud), 0 = in cloud
    ichimoku_trend = np.where(close_1d_aligned > cloud_top, 1,
                              np.where(close_1d_aligned < cloud_bottom, -1, 0))
    
    # Calculate 6h Donchian(20) channels
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 30-bar average (on 6h data)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup
        # Get current values
        donchian_high = period20_high[i]
        donchian_low = period20_low[i]
        tenkan = tenkan_sen_aligned[i]
        kijun = kijun_sen_aligned[i]
        trend = ichimoku_trend[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(donchian_high) or np.isnan(donchian_low) or np.isnan(tenkan) or np.isnan(kijun) or np.isnan(trend):
            continue
            
        # Entry conditions
        # Long: price breaks above Donchian high AND 1d Ichimoku bullish AND TK cross bullish AND volume spike
        long_entry = (close[i] > donchian_high) and (trend == 1) and (tenkan > kijun) and vol_spike
        # Short: price breaks below Donchian low AND 1d Ichimoku bearish AND TK cross bearish AND volume spike
        short_entry = (close[i] < donchian_low) and (trend == -1) and (tenkan < kijun) and vol_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR TK cross turns bearish
            if close[i] < donchian_low or tenkan < kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR TK cross turns bullish
            if close[i] > donchian_high or tenkan > kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals