#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud + 1d trend filter + volume confirmation
# Tenkan-sen (9-period) + Kijun-sen (26-period) cross + price above/below Kumo (cloud)
# Senkou Span A/B calculated from 26 periods ahead
# Long when Tenkan > Kijun, price above cloud, 1d uptrend
# Short when Tenkan < Kijun, price below cloud, 1d downtrend
# Volume confirmation: current volume > 1.5x 20-period average
# Ichimoku provides dynamic support/resistance via cloud, reducing whipsaw
# Effective in trending markets with clear trend/cloud relationship
# Targets 50-150 total trades over 4 years (12-37/year) for optimal fee drag

name = "6h_IchimokuCloud_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 52:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Ichimoku components
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_9 = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_9 = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_9 + min_low_9) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_26 = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_26 = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_26 + min_low_26) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    max_high_52 = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_52 = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (max_high_52 + min_low_52) / 2
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Need enough data for Senkou B (52-period)
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or 
            np.isnan(senkou_b[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_conf_val = vol_conf[i]
        
        # Cloud top and bottom
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        
        # Ichimoku signals
        tenkan_above_kijun = tenkan_val > kijun_val
        tenkan_below_kijun = tenkan_val < kijun_val
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        if position == 0:
            # Enter long: Tenkan > Kijun, price above cloud, 1d uptrend, volume confirmation
            if tenkan_above_kijun and price_above_cloud and ema50_1d_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Tenkan < Kijun, price below cloud, 1d downtrend, volume confirmation
            elif tenkan_below_kijun and price_below_cloud and ema50_1d_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Tenkan < Kijun or price below cloud or 1d trend down
            if tenkan_below_kijun or not price_above_cloud or ema50_1d_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Tenkan > Kijun or price above cloud or 1d trend up
            if tenkan_above_kijun or not price_below_cloud or ema50_1d_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals