#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 12h trend filter and volume confirmation
# Uses Ichimoku components (Tenkan-sen, Kijun-sen, Senkou Span A/B) from 6h timeframe
# 12h EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume spike (>1.8 * 20-period EMA) confirms institutional participation
# Cloud acts as dynamic support/resistance: price above cloud = bull bias, below = bear bias
# TK cross (Tenkan-sen crossing Kijun-sen) provides entry signals in direction of trend
# Designed for low trade frequency: ~15-25 trades/year per symbol with 0.25 sizing
# Works in bull markets via breakout above cloud and bear markets via breakdown below cloud
# BTC/ETH focused: requires 12h trend alignment and volume spike to avoid SOL-only bias

name = "6h_Ichimoku_Cloud_TK_Cross_12hEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Ichimoku Cloud calculation (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((high_52 + low_52) / 2)
    
    # Current cloud boundaries (Senkou Span A/B shifted back 26 periods to align with price)
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_lagged, senkou_b_lagged)
    cloud_bottom = np.minimum(senkou_a_lagged, senkou_b_lagged)
    
    # TK Cross: Tenkan-sen crossing above/below Kijun-sen
    tk_cross_up = (tenkan_sen > kijun_sen) & (np.roll(tenkan_sen, 1) <= np.roll(kijun_sen, 1))
    tk_cross_down = (tenkan_sen < kijun_sen) & (np.roll(tenkan_sen, 1) >= np.roll(kijun_sen, 1))
    
    # Volume confirmation: volume > 1.8 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for Ichimoku (52+26=78 periods)
    start_idx = 80
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(tenkan_sen[i]) or 
            np.isnan(kijun_sen[i]) or np.isnan(cloud_top[i]) or 
            np.isnan(cloud_bottom[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h EMA50
        bullish_bias = close[i] > ema_50_12h_aligned[i]
        bearish_bias = close[i] < ema_50_12h_aligned[i]
        
        # Determine price position relative to cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias and price_above_cloud:
                # Long: TK cross up in bullish alignment with volume spike
                if tk_cross_up[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias and price_below_cloud:
                # Short: TK cross down in bearish alignment with volume spike
                if tk_cross_down[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # No clear signal
        
        elif position == 1:  # Long position
            # Exit: price breaks below cloud bottom or TK cross down
            if close[i] < cloud_bottom[i] or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above cloud top or TK cross up
            if close[i] > cloud_top[i] or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals