#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1w trend filter and 1d volume confirmation
# - Entry: Long when price > Kumo cloud + Tenkan > Kijun (bullish TK cross) + 1w price > Kumo (long-term uptrend) + 1d volume > 1.5x 20-period average
#          Short when price < Kumo cloud + Tenkan < Kijun (bearish TK cross) + 1w price < Kumo (long-term downtrend) + 1d volume > 1.5x 20-period average
# - Exit: Close-based reversal - exit long when price < Kumo cloud, exit short when price > Kumo cloud
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Ichimoku components from 6h data for entry signals, weekly Ichimoku for trend filter, daily volume for confirmation
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within HARD MAX: 300 total
# - Ichimoku provides robust trend/filter signals that work in both bull and bear markets via cloud positioning
# - Volume confirmation ensures genuine participation, reducing false signals
# - Weekly trend filter aligns with major market regime, avoiding counter-trend trades

name = "6h_1w_1d_ichimoku_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Pre-compute 1w data for Ichimoku
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pre-compute 1d data for volume
    volume_1d = df_1d['volume'].values
    
    # Calculate 6h Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high_6h = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low_6h = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan_6h = (period9_high_6h + period9_low_6h) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high_6h = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low_6h = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun_6h = (period26_high_6h + period26_low_6h) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a_6h = (tenkan_6h + kijun_6h) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high_6h = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low_6h = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b_6h = (period52_high_6h + period52_low_6h) / 2
    
    # Kumo cloud boundaries
    senkou_top_6h = np.maximum(senkou_a_6h, senkou_b_6h)
    senkou_bottom_6h = np.minimum(senkou_a_6h, senkou_b_6h)
    
    # Calculate 1w Ichimoku for trend filter
    period9_high_1w = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low_1w = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_1w = (period9_high_1w + period9_low_1w) / 2
    
    period26_high_1w = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low_1w = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_1w = (period26_high_1w + period26_low_1w) / 2
    
    senkou_a_1w = (tenkan_1w + kijun_1w) / 2
    
    period52_high_1w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low_1w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = (period52_high_1w + period52_low_1w) / 2
    
    senkou_top_1w = np.maximum(senkou_a_1w, senkou_b_1w)
    senkou_bottom_1w = np.minimum(senkou_a_1w, senkou_b_1w)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 6h timeframe
    tenkan_6h_aligned = align_htf_to_ltf(prices, prices, tenkan_6h)  # 6h data, no alignment needed
    kijun_6h_aligned = align_htf_to_ltf(prices, prices, kijun_6h)
    senkou_top_6h_aligned = align_htf_to_ltf(prices, prices, senkou_top_6h)
    senkou_bottom_6h_aligned = align_htf_to_ltf(prices, prices, senkou_bottom_6h)
    
    tenkan_1w_aligned = align_htf_to_ltf(prices, df_1w, tenkan_1w)
    kijun_1w_aligned = align_htf_to_ltf(prices, df_1w, kijun_1w)
    senkou_top_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_top_1w)
    senkou_bottom_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_bottom_1w)
    
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(52, n):  # Start after warmup period (max Ichimoku period)
        # Skip if any required data is invalid
        if (np.isnan(tenkan_6h_aligned[i]) or np.isnan(kijun_6h_aligned[i]) or 
            np.isnan(senkou_top_6h_aligned[i]) or np.isnan(senkou_bottom_6h_aligned[i]) or
            np.isnan(tenkan_1w_aligned[i]) or np.isnan(kijun_1w_aligned[i]) or
            np.isnan(senkou_top_1w_aligned[i]) or np.isnan(senkou_bottom_1w_aligned[i]) or
            np.isnan(volume_ma_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current prices
        close_price = close_6h[i]
        
        # 6h Ichimoku conditions
        price_above_kumo = close_price > senkou_top_6h_aligned[i]
        price_below_kumo = close_price < senkou_bottom_6h_aligned[i]
        tk_bullish = tenkan_6h_aligned[i] > kijun_6h_aligned[i]
        tk_bearish = tenkan_6h_aligned[i] < kijun_6h_aligned[i]
        
        # 1w Ichimoku trend filter
        weekly_bullish = close_1w[-1] > senkou_top_1w_aligned[i] if len(close_1w) > 0 else False  # Simplified: use current week's close vs cloud
        weekly_bearish = close_1w[-1] < senkou_bottom_1w_aligned[i] if len(close_1w) > 0 else False
        
        # Volume confirmation
        volume_confirmation = volume_1d_aligned[i] > 1.5 * volume_ma_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Kumo + TK bullish + weekly bullish + volume confirmation
            if (price_above_kumo and tk_bullish and 
                # Use simplified weekly trend: price above weekly Kumo
                close_6h[i] > senkou_top_1w_aligned[i] and 
                volume_confirmation):
                position = 1
                signals[i] = 0.25
            # Short entry: price < Kumo + TK bearish + weekly bearish + volume confirmation
            elif (price_below_kumo and tk_bearish and 
                  # Use simplified weekly trend: price below weekly Kumo
                  close_6h[i] < senkou_bottom_1w_aligned[i] and 
                  volume_confirmation):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price < Kumo cloud
            # Exit short when price > Kumo cloud
            if position == 1:
                if close_price < senkou_top_6h_aligned[i]:  # Exit when price falls below cloud top
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_price > senkou_bottom_6h_aligned[i]:  # Exit when price rises above cloud bottom
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals