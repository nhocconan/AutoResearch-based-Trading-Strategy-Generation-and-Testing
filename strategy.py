#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d/1w trend filter
# - Primary signal: Price above/below Ichimoku cloud on 6h with TK cross confirmation
# - HTF trend filter: 1d price > 1w EMA50 for longs, price < 1w EMA50 for shorts
# - Volume confirmation: 6h volume > 1.5x 20-period average volume
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: Close below/above cloud opposite side
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: Cloud acts as dynamic support/resistance; TK cross filters false breaks

name = "6h_1d_1w_ichimoku_cloud_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute 1d close for additional trend filter (price > 1d open bias)
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    close_gt_open_1d = close_1d > open_1d
    close_gt_open_1d_aligned = align_htf_to_ltf(prices, df_1d, close_gt_open_1d)
    
    # Pre-compute 6h volume filter
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    # Pre-compute 6h Ichimoku components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_6h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_6h).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_6h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_6h).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_6h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_6h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Kumo (Cloud): between Senkou Span A and B
    # For signal generation, we use current cloud (not shifted)
    # We need to calculate the cloud values for current period
    # Senkou Span A current = (tenkan + kijun)/2 (this is actually for current cloud)
    # Senkou Span B current = (period52_high + period52_low)/2 (this is for current cloud)
    senkou_a_current = (tenkan + kijun) / 2
    senkou_b_current = (period52_high + period52_low) / 2
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_current, senkou_b_current)
    cloud_bottom = np.minimum(senkou_a_current, senkou_b_current)
    
    # TK Cross: Tenkan crosses above/below Kijun
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # Start after Ichimoku warmup
        # Skip if any required data is invalid
        if (np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price closes below cloud OR TK cross down
            if close_6h[i] < cloud_bottom[i] or tk_cross_down[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price closes above cloud OR TK cross up
            if close_6h[i] > cloud_top[i] or tk_cross_up[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entries with all filters aligned
            if vol_spike[i]:
                # Long: Price above cloud, TK cross up, 1d close > 1d open, price > 1w EMA50
                if (close_6h[i] > cloud_top[i] and 
                    tk_cross_up[i] and 
                    close_gt_open_1d_aligned[i] and 
                    close_6h[i] > ema_50_1w_aligned[i]):
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Price below cloud, TK cross down, 1d close < 1d open, price < 1w EMA50
                elif (close_6h[i] < cloud_bottom[i] and 
                      tk_cross_down[i] and 
                      not close_gt_open_1d_aligned[i] and 
                      close_6h[i] < ema_50_1w_aligned[i]):
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals