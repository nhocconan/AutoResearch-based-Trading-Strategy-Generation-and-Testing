#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation
# Ichimoku Cloud acts as dynamic support/resistance with forward-looking Senkou Span
# Price above/below cloud indicates trend direction, TK cross provides entry timing
# 1d EMA50 filter ensures alignment with higher timeframe trend
# Volume spike (1.8x 20-period average) confirms breakout strength
# Works in bull markets via breakouts above cloud and bear markets via breakdowns below cloud
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Ichimoku_Cloud_Breakout_1dEMA50_Volume_v1"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Ichimoku Cloud components (6h timeframe)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = (high_senkou_b + low_senkou_b) / 2
    
    # Chikou Span (Lagging Span): not used for signals (would require look-ahead)
    
    # Cloud top and bottom (Senkou Span A and B shifted forward 26 periods)
    # For signal generation at time t, we use cloud values that were plotted 26 periods ago
    # So we need to shift Senkou Span A and B BACKWARD by 26 to align with current price
    displacement = 26
    senkou_span_a_lagged = np.roll(senkou_span_a, displacement)
    senkou_span_b_lagged = np.roll(senkou_span_b, displacement)
    # Fill the first 'displacement' values with NaN (no cloud data available yet)
    senkou_span_a_lagged[:displacement] = np.nan
    senkou_span_b_lagged[:displacement] = np.nan
    
    # Cloud top is the higher of Senkou Span A and B
    # Cloud bottom is the lower of Senkou Span A and B
    cloud_top = np.maximum(senkou_span_a_lagged, senkou_span_b_lagged)
    cloud_bottom = np.minimum(senkou_span_a_lagged, senkou_span_b_lagged)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Tenkan(9), Kijun(26), Senkou B(52), displacement(26), EMA50(50), vol MA(20)
    start_idx = max(9, 26, 52, 26, 50, 20)  # = 52
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_tenkan = tenkan_sen[i]
        curr_kijun = kijun_sen[i]
        curr_cloud_top = cloud_top[i]
        curr_cloud_bottom = cloud_bottom[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Determine if price is above or below cloud
        price_above_cloud = curr_close > curr_cloud_top
        price_below_cloud = curr_close < curr_cloud_bottom
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above cloud AND above 1d EMA50 (uptrend)
                # Plus Tenkan-sen > Kijun-sen (bullish momentum)
                if price_above_cloud and curr_close > curr_ema_50_1d and curr_tenkan > curr_kijun:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below cloud AND below 1d EMA50 (downtrend)
                # Plus Tenkan-sen < Kijun-sen (bearish momentum)
                elif price_below_cloud and curr_close < curr_ema_50_1d and curr_tenkan < curr_kijun:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price falls below cloud bottom OR Tenkan-sen < Kijun-sen (momentum loss)
            if curr_close < curr_cloud_bottom or curr_tenkan < curr_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above cloud top OR Tenkan-sen > Kijun-sen (momentum loss)
            if curr_close > curr_cloud_top or curr_tenkan > curr_kijun:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals