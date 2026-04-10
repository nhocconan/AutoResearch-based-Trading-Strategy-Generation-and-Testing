#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d/1w filters for BTC/ETH
# - Primary: 6h Ichimoku system (Tenkan/Kijun cross, price vs Cloud)
# - HTF: 1d trend filter (price above/below 1d Kumo), 1w volume confirmation
# - Long: Tenkan > Kijun + price > 6h Cloud + price > 1d Cloud + 1w volume > 1.2x 4w MA
# - Short: Tenkan < Kijun + price < 6h Cloud + price < 1d Cloud + 1w volume > 1.2x 4w MA
# - Exit: Tenkan/Kijun cross reversal OR price crosses 6h Cloud
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Ichimoku adapts to volatility, volume confirms conviction, multi-timeframe alignment reduces false signals

name = "6h_1d_1w_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for Ichimoku calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pre-compute 1w data
    volume_1w = df_1w['volume'].values
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = np.full(n, np.nan)
    min_low_9 = np.full(n, np.nan)
    for i in range(period_tenkan - 1, n):
        if not np.isnan(high[i-period_tenkan+1:i+1]).any() and not np.isnan(low[i-period_tenkan+1:i+1]).any():
            max_high_9[i] = np.max(high[i-period_tenkan+1:i+1])
            min_low_9[i] = np.min(low[i-period_tenkan+1:i+1])
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = np.full(n, np.nan)
    min_low_26 = np.full(n, np.nan)
    for i in range(period_kijun - 1, n):
        if not np.isnan(high[i-period_kijun+1:i+1]).any() and not np.isnan(low[i-period_kijun+1:i+1]).any():
            max_high_26[i] = np.max(high[i-period_kijun+1:i+1])
            min_low_26[i] = np.min(low[i-period_kijun+1:i+1])
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 plotted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = np.full(n, np.nan)
    min_low_52 = np.full(n, np.nan)
    for i in range(period_senkou_b - 1, n):
        if not np.isnan(high[i-period_senkou_b+1:i+1]).any() and not np.isnan(low[i-period_senkou_b+1:i+1]).any():
            max_high_52[i] = np.max(high[i-period_senkou_b+1:i+1])
            min_low_52[i] = np.min(low[i-period_senkou_b+1:i+1])
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Calculate 6h Cloud (Kumo): between Senkou Span A and B
    # For plotting, these are shifted forward 26 periods, so we use current values
    # We'll consider price above/below current cloud boundaries
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Calculate 1d Ichimoku Cloud for trend filter
    period_tenkan_1d = 9
    period_kijun_1d = 26
    period_senkou_b_1d = 52
    
    max_high_9_1d = np.full(len(high_1d), np.nan)
    min_low_9_1d = np.full(len(low_1d), np.nan)
    for i in range(period_tenkan_1d - 1, len(high_1d)):
        if not np.isnan(high_1d[i-period_tenkan_1d+1:i+1]).any() and not np.isnan(low_1d[i-period_tenkan_1d+1:i+1]).any():
            max_high_9_1d[i] = np.max(high_1d[i-period_tenkan_1d+1:i+1])
            min_low_9_1d[i] = np.min(low_1d[i-period_tenkan_1d+1:i+1])
    tenkan_1d = (max_high_9_1d + min_low_9_1d) / 2
    
    max_high_26_1d = np.full(len(high_1d), np.nan)
    min_low_26_1d = np.full(len(low_1d), np.nan)
    for i in range(period_kijun_1d - 1, len(high_1d)):
        if not np.isnan(high_1d[i-period_kijun_1d+1:i+1]).any() and not np.isnan(low_1d[i-period_kijun_1d+1:i+1]).any():
            max_high_26_1d[i] = np.max(high_1d[i-period_kijun_1d+1:i+1])
            min_low_26_1d[i] = np.min(low_1d[i-period_kijun_1d+1:i+1])
    kijun_1d = (max_high_26_1d + min_low_26_1d) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    max_high_52_1d = np.full(len(high_1d), np.nan)
    min_low_52_1d = np.full(len(low_1d), np.nan)
    for i in range(period_senkou_b_1d - 1, len(high_1d)):
        if not np.isnan(high_1d[i-period_senkou_b_1d+1:i+1]).any() and not np.isnan(low_1d[i-period_senkou_b_1d+1:i+1]).any():
            max_high_52_1d[i] = np.max(high_1d[i-period_senkou_b_1d+1:i+1])
            min_low_52_1d[i] = np.min(low_1d[i-period_senkou_b_1d+1:i+1])
    senkou_b_1d = (max_high_52_1d + min_low_52_1d) / 2
    
    cloud_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    cloud_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Calculate 1w volume moving average (4-period) for volume confirmation
    volume_ma_4_1w = np.full(len(volume_1w), np.nan)
    for i in range(3, len(volume_1w)):
        if not np.isnan(volume_1w[i-3:i+1]).any():
            volume_ma_4_1w[i] = np.mean(volume_1w[i-3:i+1])
    
    # Align all HTF indicators to 6h timeframe
    cloud_top_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_top_1d)
    cloud_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, cloud_bottom_1d)
    volume_ma_4_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_4_1w)
    volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after all indicators are calculated (need 52 periods for Senkou B)
    start_idx = max(period_tenkan, period_kijun, period_senkou_b) + 26  # +26 for cloud lookahead
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(cloud_top_1d_aligned[i]) or np.isnan(cloud_bottom_1d_aligned[i]) or
            np.isnan(volume_ma_4_1w_aligned[i]) or np.isnan(volume_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Ichimoku signals
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        price_above_6h_cloud = close[i] > cloud_top[i]
        price_below_6h_cloud = close[i] < cloud_bottom[i]
        
        # 1d trend filter: price relative to 1d cloud
        price_above_1d_cloud = close[i] > cloud_top_1d_aligned[i]
        price_below_1d_cloud = close[i] < cloud_bottom_1d_aligned[i]
        
        # Volume confirmation: current 1w volume > 1.2x 4-period MA
        volume_confirm = volume_1w_aligned[i] > 1.2 * volume_ma_4_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Tenkan > Kijun + price > 6h Cloud + price > 1d Cloud + volume confirmation
            if (tenkan_above_kijun and price_above_6h_cloud and 
                price_above_1d_cloud and volume_confirm):
                position = 1
                signals[i] = 0.25
            # Short entry: Tenkan < Kijun + price < 6h Cloud + price < 1d Cloud + volume confirmation
            elif (tenkan_below_kijun and price_below_6h_cloud and 
                  price_below_1d_cloud and volume_confirm):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Tenkan/Kijun cross reversal OR price crosses 6h Cloud
            if position == 1:  # Long position
                exit_condition = (tenkan_below_kijun or price_below_6h_cloud)
            else:  # position == -1 (Short position)
                exit_condition = (tenkan_above_kijun or price_above_6h_cloud)
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals