#!/usr/bin/env python3
"""
6h Ichimoku Cloud Breakout with 1d Trend Filter and Volume Confirmation
Hypothesis: Ichimoku TK cross above/below cloud on 6h, aligned with 1d EMA50 trend and confirmed by volume spikes,
captures strong momentum moves. Ichimoku provides dynamic support/resistance (cloud) and momentum (TK cross),
working in both bull and bear markets by filtering with higher timeframe trend. 6h timeframe targets 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (high_tenkan + low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (high_kijun + low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
    period_senkou_b = 52
    high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((high_senkou_b + low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): Close plotted 26 periods behind (not used for signals due to look-ahead)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Ichimoku calculations (max period is 52 for Senkou B)
    start_idx = max(period_senkou_b, 50, 20)  # Senkou B + EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Cloud top and bottom (Senkou Span A and B)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        
        # TK Cross: Tenkan-sen crossing Kijun-sen
        tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
        
        # Price above/below cloud
        price_above_cloud = curr_close > cloud_top
        price_below_cloud = curr_close < cloud_bottom
        
        # Daily trend filter: price above/below EMA50
        uptrend = ema_50_aligned[i] is not None and curr_close > ema_50_aligned[i]
        downtrend = ema_50_aligned[i] is not None and curr_close < ema_50_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: TK cross up AND price above cloud AND uptrend AND volume spike
            long_entry = tk_cross_up and price_above_cloud and uptrend and vol_spike
            # Short: TK cross down AND price below cloud AND downtrend AND volume spike
            short_entry = tk_cross_down and price_below_cloud and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below cloud OR loss of trend (price < EMA50) OR TK cross down
            if (curr_close < cloud_bottom) or (curr_close < ema_50_aligned[i]) or tk_cross_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above cloud OR loss of trend (price > EMA50) OR TK cross up
            if (curr_close > cloud_top) or (curr_close > ema_50_aligned[i]) or tk_cross_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_CloudBreakout_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0