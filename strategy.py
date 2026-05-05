#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when price breaks above Ichimoku Cloud (Senkou Span A/B) AND Tenkan > Kijun AND 1d EMA50 uptrend AND volume > 1.5x 20-period average
# Short when price breaks below Ichimoku Cloud AND Tenkan < Kijun AND 1d EMA50 downtrend AND volume > 1.5x 20-period average
# Exit when price re-enters the cloud OR 1d trend reverses
# Ichimoku provides dynamic support/resistance, cloud acts as trend filter, Tenkan/Kijun cross confirms momentum,
# 1d EMA50 avoids counter-trend trades, volume spike filters for institutional participation.
# Works in bull markets via longs above cloud and bear markets via shorts below cloud.

name = "6h_Ichimoku_CloudBreakout_1dEMA50_Volume"
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
    
    # Get 6h data ONCE before loop for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 52:  # Need at least 52 periods for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(df_6h['high']).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(df_6h['low']).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(df_6h['high']).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(df_6h['low']).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(df_6h['high']).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(df_6h['low']).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h timeframe (they are already on 6h)
    tenkan_aligned = tenkan  # Already on 6h timeframe
    kijun_aligned = kijun
    senkou_a_aligned = senkou_a
    senkou_b_aligned = senkou_b
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 1d trend to 6h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Determine cloud boundaries (upper and lower bands of the cloud)
    upper_band = np.maximum(senkou_a_aligned, senkou_b_aligned)
    lower_band = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after Ichimoku warmup
        # Skip if any value is NaN
        if (np.isnan(tenkan_aligned[i]) or 
            np.isnan(kijun_aligned[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or 
            np.isnan(downtrend_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above cloud AND Tenkan > Kijun AND 1d EMA50 uptrend AND volume spike
            if (close[i] > upper_band[i] and 
                tenkan_aligned[i] > kijun_aligned[i] and 
                uptrend_1d_aligned[i] > 0.5 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below cloud AND Tenkan < Kijun AND 1d EMA50 downtrend AND volume spike
            elif (close[i] < lower_band[i] and 
                  tenkan_aligned[i] < kijun_aligned[i] and 
                  downtrend_1d_aligned[i] > 0.5 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters cloud (below upper band) OR 1d trend changes to downtrend
            if (close[i] < upper_band[i] or 
                downtrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters cloud (above lower band) OR 1d trend changes to uptrend
            if (close[i] > lower_band[i] or 
                uptrend_1d_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals