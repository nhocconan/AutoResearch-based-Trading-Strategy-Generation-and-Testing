#!/usr/bin/env python3
"""
exp_6575_6h_ichimoku_cloud_1w_trend_v1
Hypothesis: 6h Ichimoku cloud with weekly trend filter for institutional trend following.
Uses weekly Ichimoku cloud (from 1d resampled to weekly) as trend filter: price above cloud = bullish bias,
price below cloud = bearish bias. 6h Tenkan/Kijun cross provides entry timing with volume confirmation.
Ichimoku works in both bull/bear markets: in bull, cloud acts as dynamic support; in bear, as resistance.
Weekly filter reduces whipsaws by ensuring alignment with major trend. Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6575_6h_ichimoku_cloud_1w_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9      # 6h periods
KIJUN_PERIOD = 26      # 6h periods
SENKOU_PERIOD = 52     # 6h periods
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
MAX_HOLD_BARS = 48     # Max hold: ~12 days (48 * 6h = 12 days)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly HTF data ONCE before loop (from 1d data)
    df_1d = get_htf_data(prices, '1d')
    
    # Resample 1d to weekly manually for Ichimoku (using actual weekly boundaries)
    # Create weekly OHLC from daily data
    weekly_high = []
    weekly_low = []
    weekly_close = []
    
    # Group by week (using resample-like logic but on actual data)
    # We'll compute weekly Ichimoku by looking back 5 days
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Weekly Tenkan/Kijun/Senkou (using 1d data, 5d and 10d periods)
    # Weekly Tenkan: 9-period high/low of daily (approx 1.5 weeks)
    # Weekly Kijun: 26-period high/low of daily (approx 1 month)
    # For simplicity, we use daily data to approximate weekly Ichimoku
    # Tenkan_1w: (9-period high + 9-period low)/2 of daily
    tenkan_1d = (pd.Series(high_1d).rolling(window=9, min_periods=9).max() + 
                 pd.Series(low_1d).rolling(window=9, min_periods=9).min()) / 2
    # Kijun_1w: (26-period high + 26-period low)/2 of daily
    kijun_1d = (pd.Series(high_1d).rolling(window=26, min_periods=26).max() + 
                pd.Series(low_1d).rolling(window=26, min_periods=26).min()) / 2
    # Senkou A: (Tenkan + Kijun)/2 shifted
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    # Senkou B: (52-period high + 52-period low)/2 shifted
    senkou_b_1d = (pd.Series(high_1d).rolling(window=52, min_periods=52).max() + 
                   pd.Series(low_1d).rolling(window=52, min_periods=52).min()) / 2
    
    # Cloud: Senkou A and Senkou B
    # Shift by 26 periods for Ichimoku cloud (but we'll use current for trend filter)
    # For trend: price above/both Senkou lines = bullish, below both = bearish
    weekly_bullish = (senkou_a_1d > senkou_b_1d) & (close_1d > senkou_a_1d) & (close_1d > senkou_b_1d)
    weekly_bearish = (senkou_a_1d < senkou_b_1d) & (close_1d < senkou_a_1d) & (close_1d < senkou_b_1d)
    
    # Align weekly trend to 6h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish.astype(float))
    
    # Calculate 6h Ichimoku components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    senkou_b = (pd.Series(high).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_PERIOD, min_periods=SENKOU_PERIOD).min()) / 2
    
    # Current cloud (Senkou A and B, not shifted for current price comparison)
    # For trend detection: price above cloud = bullish, below cloud = bearish
    # Cloud top = max(Senkou A, Senkou B), cloud bottom = min(Senkou A, Senkou B)
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Price relative to cloud
    price_above_cloud = close > cloud_top
    price_below_cloud = close < cloud_bottom
    price_in_cloud = ~(price_above_cloud | price_below_cloud)
    
    # Tenkan/Kijun cross
    tenkan_prev = np.roll(tenkan, 1)
    kijun_prev = np.roll(kijun, 1)
    tenkan_prev[0] = np.nan
    kijun_prev[0] = np.nan
    
    tk_cross_up = (tenkan > kijun) & (tenkan_prev <= kijun_prev)  # Bullish cross
    tk_cross_down = (tenkan < kijun) & (tenkan_prev >= kijun_prev)  # Bearish cross
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_PERIOD, VOL_MA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            # Exit if: price drops below cloud OR Tenkan/Kijun bearish cross
            exit_long = price_below_cloud[i] or tk_cross_down[i]
            # Time-based exit
            exit_long = exit_long or bars_since_entry >= MAX_HOLD_BARS
            if exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            # Exit if: price rises above cloud OR Tenkan/Kijun bullish cross
            exit_short = price_above_cloud[i] or tk_cross_up[i]
            # Time-based exit
            exit_short = exit_short or bars_since_entry >= MAX_HOLD_BARS
            if exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        
        # Enter new positions only if flat
        if position == 0:
            # Long conditions:
            # 1. Bullish Tenkan/Kijun cross
            # 2. Price above 6h cloud (or at least not below)
            # 3. Weekly trend bullish OR price above weekly cloud (flexible)
            # 4. Volume confirmation
            long_cross = tk_cross_up[i]
            long_price = price_above_cloud[i] or (not price_below_cloud[i] and close[i] > senkou_a[i])  # Above or in cloud
            long_weekly = weekly_bullish_aligned[i] > 0.5  # Weekly bullish
            long_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
            
            # Short conditions:
            # 1. Bearish Tenkan/Kijun cross
            # 2. Price below 6h cloud (or at least not above)
            # 3. Weekly trend bearish OR price below weekly cloud
            # 4. Volume confirmation
            short_cross = tk_cross_down[i]
            short_price = price_below_cloud[i] or (not price_above_cloud[i] and close[i] < senkou_b[i])  # Below or in cloud
            short_weekly = weekly_bearish_aligned[i] > 0.5  # Weekly bearish
            short_volume = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
            
            if long_cross and long_price and (long_weekly or long_price) and long_volume:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_cross and short_price and (short_weekly or short_price) and short_volume:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals