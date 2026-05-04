#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d weekly pivot direction filter and volume confirmation (>1.2x 20 EMA)
# Uses Ichimoku components from prior completed 6h bar for trend/cloud signals, 1d weekly pivot for higher timeframe bias
# Volume confirmation ensures signal has strong participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# Ichimoku cloud acts as dynamic support/resistance, weekly pivot provides structural bias, volume filters weak breakouts.
# Works in bull (cloud support, pivot longs) and bear (cloud resistance, pivot shorts) via clear trend definition.

name = "6h_Ichimoku_1dWeeklyPivot_VolumeConfirm"
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
    
    # Get 1d data for weekly pivot (using prior week's H/L/C)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot from 1d data (prior completed week)
    # Group into weeks: week_start Monday, week_end Sunday
    # Use prior completed week's high, low, close for pivot
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly high/low/close: resample logic via rolling window of 5 days (approx)
    # Using min_periods=5 to ensure full week
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot levels (standard formula)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot, additional_delay_bars=0)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1, additional_delay_bars=0)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1, additional_delay_bars=0)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2, additional_delay_bars=0)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2, additional_delay_bars=0)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3, additional_delay_bars=0)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3, additional_delay_bars=0)
    
    # Ichimoku components (using prior completed 6h bar)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2
    
    # Align Ichimoku components to current time (no look-ahead)
    # Tenkan and Kijun: use current values (no lag needed as they're based on past)
    tenkan_aligned = tenkan  # already uses only historical data
    kijun_aligned = kijun
    # Senkou spans: need to shift forward 26 periods to align with ichimoku cloud plotting
    senkou_a_aligned = np.roll(senkou_a, 26)
    senkou_b_aligned = np.roll(senkou_b, 26)
    senkou_a_aligned[:26] = np.nan
    senkou_b_aligned[:26] = np.nan
    
    # Current price vs cloud: price above cloud = bullish, below = bearish
    # Cloud top = max(senkou_a, senkou_b), cloud bottom = min(senkou_a, senkou_b)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Ichimoku signals:
        # TK Cross: Tenkan crosses above/below Kijun
        tk_cross_above = tenkan_aligned[i] > kijun_aligned[i] and tenkan_aligned[i-1] <= kijun_aligned[i-1]
        tk_cross_below = tenkan_aligned[i] < kijun_aligned[i] and tenkan_aligned[i-1] >= kijun_aligned[i-1]
        
        # Price vs Cloud
        price_above_cloud = close[i] > cloud_top[i]
        price_below_cloud = close[i] < cloud_bottom[i]
        
        if position == 0:
            # Long conditions: TK cross above + price above cloud + price above weekly pivot + volume spike
            if (tk_cross_above and price_above_cloud and 
                close[i] > weekly_pivot_aligned[i] and 
                volume[i] > (1.2 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: TK cross below + price below cloud + price below weekly pivot + volume spike
            elif (tk_cross_below and price_below_cloud and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume[i] > (1.2 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TK cross below OR price falls below cloud bottom OR price falls below weekly S1
            if (tk_cross_below or close[i] < cloud_bottom[i] or 
                close[i] < weekly_s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TK cross above OR price rises above cloud top OR price rises above weekly R1
            if (tk_cross_above or close[i] > cloud_top[i] or 
                close[i] > weekly_r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals