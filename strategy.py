#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeFilter
Hypothesis: On 6h chart, enter long when price breaks above Kumo (cloud) with TK cross bullish and 1w trend up (price > 1w EMA50), short when price breaks below Kumo with TK cross bearish and 1w trend down. Volume confirmation (>1.5x 20-bar MA) filters weak breakouts. Works in bull/bear via 1w trend filter. Uses discrete sizing (0.25) to minimize fee churn. Targets 12-37 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_tenkan + min_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_kijun + min_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_senkou_b + min_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    # Not used for entry, but can be used for confirmation if needed
    
    # The cloud (Kumo) is between Senkou Span A and Senkou Span B
    # For breakout signals, we compare current price to the cloud
    # Since Senkou spans are shifted 26 periods ahead, we need to align them
    # For simplicity, we use the current cloud (unshifted) for breakout detection
    # This is acceptable as we're looking for price to be above/below current cloud
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations needed
    start_idx = max(period_tenkan, period_kijun, period_senkou_b, 26) + 26  # +26 for Senkou shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Kumo (cloud) boundaries
        upper_cloud = max(senkou_a_val, senkou_b_val)
        lower_cloud = min(senkou_a_val, senkou_b_val)
        
        # TK cross
        tk_bullish = tenkan_val > kijun_val
        tk_bearish = tenkan_val < kijun_val
        
        # 1w trend filter
        trend_up = close_val > ema_50_1w_val
        trend_down = close_val < ema_50_1w_val
        
        # Entry conditions
        # Long: price above cloud, TK bullish, 1w trend up, volume spike
        long_entry = (close_val > upper_cloud) and tk_bullish and trend_up and vol_spike
        # Short: price below cloud, TK bearish, 1w trend down, volume spike
        short_entry = (close_val < lower_cloud) and tk_bearish and trend_down and vol_spike
        
        # Exit conditions: opposite TK cross or price re-enters cloud
        exit_long = tk_bearish or (close_val < upper_cloud and close_val > lower_cloud)
        exit_short = tk_bullish or (close_val < upper_cloud and close_val > lower_cloud)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0