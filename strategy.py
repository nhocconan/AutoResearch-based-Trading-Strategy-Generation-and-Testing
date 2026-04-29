#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Long when Tenkan crosses above Kijun AND price above cloud AND price > 1d EMA50 AND volume > 1.5x 24-bar avg
# Short when Tenkan crosses below Kijun AND price below cloud AND price < 1d EMA50 AND volume > 1.5x 24-bar avg
# Exit when Tenkan/Kijun cross reverses OR price touches opposite cloud edge
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-25 trades/year on 6h.
# Ichimoku provides dynamic support/resistance with trend, momentum, and volatility in one system.
# 1d EMA50 filter ensures we only trade with the higher timeframe trend, improving win rate in bear markets.
# Volume confirmation ensures breakouts have conviction, reducing false signals in choppy markets.

name = "6h_IchimokuTK_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2.0
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (high_senkou_b + low_senkou_b) / 2.0
    
    # Volume confirmation: >1.5x 24-bar average volume (4 periods on 6h = 1 day)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.5 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 52)  # Need Senkou B (52-period) and volume MA (24-period)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        tenkan_i = tenkan[i]
        kijun_i = kijun[i]
        senkou_a_i = senkou_a[i]
        senkou_b_i = senkou_b[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = max(senkou_a_i, senkou_b_i)
        lower_cloud = min(senkou_a_i, senkou_b_i)
        
        # Check for Tenkan/Kijun cross
        tenkan_prev = tenkan[i-1] if i > 0 else tenkan_i
        kijun_prev = kijun[i-1] if i > 0 else kijun_i
        tk_cross_above = tenkan_i > kijun_i and tenkan_prev <= kijun_prev
        tk_cross_below = tenkan_i < kijun_i and tenkan_prev >= kijun_prev
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Tenkan crosses above Kijun AND price above cloud AND price > 1d EMA50 AND volume confirmation
            if tk_cross_above and curr_close > upper_cloud and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Tenkan crosses below Kijun AND price below cloud AND price < 1d EMA50 AND volume confirmation
            elif tk_cross_below and curr_close < lower_cloud and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Tenkan/Kijun cross reverses OR price touches lower cloud
            if tk_cross_below or curr_low < lower_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Tenkan/Kijun cross reverses OR price touches upper cloud
            if tk_cross_above or curr_high > upper_cloud:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals