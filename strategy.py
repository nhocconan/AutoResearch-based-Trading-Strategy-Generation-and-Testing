#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud breakout with 1d trend filter and volume confirmation.
# Enter long when Tenkan-sen crosses above Kijun-sen and price is above the cloud (Senkou Span A/B),
# with 1d EMA50 uptrend and volume > 1.5x 20-bar average. Enter short on opposite conditions.
# Exit when price re-enters the cloud or Tenkan/Kijun cross reverses.
# Ichimoku provides dynamic support/resistance and trend identification. The cloud acts as a
# trend filter that works in both bull and bear markets by adapting to volatility.
# Using 6h timeframe with 1d HTF for trend alignment reduces noise while capturing medium-term moves.

name = "6h_Ichimoku_CloudBreakout_1dTrend_Volume_v1"
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
    
    # Get 6h data for Ichimoku calculation
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 50:  # Need sufficient data for Ichimoku
        return np.zeros(n)
    
    # Calculate Ichimoku components on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # Align Ichimoku components to 6h (shifted to avoid look-ahead)
    tenkan_aligned = align_htf_to_ltf(prices, df_6h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_6h, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_6h, senkou_span_b)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:  # Need sufficient data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA (50-period)
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for Ichimoku and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(senkou_a_aligned[i]) or np.isnan(senkou_b_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA trend filter: price > EMA50 = uptrend, price < EMA50 = downtrend
        ema_trend_up = close[i] > ema_50_aligned[i]
        ema_trend_down = close[i] < ema_50_aligned[i]
        
        # Ichimoku conditions
        price = close[i]
        tenkan = tenkan_aligned[i]
        kijun = kijun_aligned[i]
        span_a = senkou_a_aligned[i]
        span_b = senkou_b_aligned[i]
        
        # Cloud top and bottom
        cloud_top = max(span_a, span_b)
        cloud_bottom = min(span_a, span_b)
        
        # Tenkan/Kijun cross
        tk_cross_up = tenkan > kijun
        tk_cross_down = tenkan < kijun
        
        # Price relative to cloud
        price_above_cloud = price > cloud_top
        price_below_cloud = price < cloud_bottom
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: TK cross up, price above cloud, uptrend, volume confirm
            if tk_cross_up and price_above_cloud and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: TK cross down, price below cloud, downtrend, volume confirm
            elif tk_cross_down and price_below_cloud and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit conditions
            # Exit: price re-enters cloud OR TK cross down
            if price <= cloud_top or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit conditions
            # Exit: price re-enters cloud OR TK cross up
            if price >= cloud_bottom or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals