#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_1dCloud_Filter_VolumeConfirm
Hypothesis: Ichimoku TK cross on 6h with 1d cloud filter (price above/below Kumo) and volume confirmation (>1.5x 20-bar avg) captures strong momentum while avoiding false signals in chop. Uses ATR(14) trailing stop (2.0) and discrete sizing (0.25). Designed to work in bull/bear via cloud filter and trend confirmation. Targets 15-25 trades/year.
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
    
    # Get 1d data for HTF cloud filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind (not used for signals)
    
    # Get 1d cloud data for filter
    # Senkou Span A on 1d
    max_high_tenkan_1d = pd.Series(high_1d).rolling(window=9, min_periods=9).max().values
    min_low_tenkan_1d = pd.Series(low_1d).rolling(window=9, min_periods=9).min().values
    tenkan_1d = (max_high_tenkan_1d + min_low_tenkan_1d) / 2
    
    max_high_kijun_1d = pd.Series(high_1d).rolling(window=26, min_periods=26).max().values
    min_low_kijun_1d = pd.Series(low_1d).rolling(window=26, min_periods=26).min().values
    kijun_1d = (max_high_kijun_1d + min_low_kijun_1d) / 2
    
    senkou_a_1d = (tenkan_1d + kijun_1d) / 2
    
    max_high_senkou_b_1d = pd.Series(high_1d).rolling(window=52, min_periods=52).max().values
    min_low_senkou_b_1d = pd.Series(low_1d).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((max_high_senkou_b_1d + min_low_senkou_b_1d) / 2)
    
    # Align 1d cloud to 6h
    senkou_a_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_a_1d)
    senkou_b_1d_aligned = align_htf_to_ltf(prices, df_1d, senkou_b_1d)
    
    # The actual cloud Senkou spans are shifted forward 26 periods
    # For filtering, we use the current cloud (which is Senkou A/B from 26 periods ago)
    # So we need to shift the aligned arrays back by 26 periods to get current cloud
    # But align_htf_to_ltf already accounts for HTF bar completion, so we use as-is for current cloud
    # Actually, for cloud filter, we want the cloud that is currently forming (Senkou A/B from 26 periods ago)
    # Since we're using daily data, the cloud plotted today is based on Senkou A/B calculated 26 days ago
    # For simplicity, we'll use the current Senkou A/B as the cloud boundaries (common simplification)
    # In practice, the cloud is ahead, but for filter we can use current Senkou A/B as support/resistance
    
    # ATR(14) on 6h for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume average (20-period) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(52, 26, 9, 20, 14)  # Senkou B, Kijun, Tenkan, vol MA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(senkou_a_1d_aligned[i]) or np.isnan(senkou_b_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get values
        tenkan_val = tenkan[i]
        kijun_val = kijun[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        senkou_a_1d_val = senkou_a_1d_aligned[i]
        senkou_b_1d_val = senkou_b_1d_aligned[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Cloud boundaries (use min/max of Senkou A/B)
        top_cloud = max(senkou_a_val, senkou_b_val)
        bottom_cloud = min(senkou_a_val, senkou_b_val)
        
        # 1d cloud filter: price must be above/below 1d cloud for bias
        top_cloud_1d = max(senkou_a_1d_val, senkou_b_1d_val)
        bottom_cloud_1d = min(senkou_a_1d_val, senkou_b_1d_val)
        
        # TK cross signals
        tk_cross_up = tenkan_val > kijun_val and tenkan[i-1] <= kijun[i-1]
        tk_cross_down = tenkan_val < kijun_val and tenkan[i-1] >= kijun[i-1]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: TK cross up + price above cloud + above 1d cloud + volume
            long_signal = tk_cross_up and (close_val > top_cloud) and (close_val > top_cloud_1d) and volume_confirm
            # Short: TK cross down + price below cloud + below 1d cloud + volume
            short_signal = tk_cross_down and (close_val < bottom_cloud) and (close_val < bottom_cloud_1d) and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit conditions:
            # 1. Stoploss: price drops 2.0*ATR from highest since entry
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # 2. TK cross down (exit long)
            elif tk_cross_down:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit conditions:
            # 1. Stoploss: price rises 2.0*ATR from lowest since entry
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # 2. TK cross up (exit short)
            elif tk_cross_up:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "6h_Ichimoku_TK_Cross_1dCloud_Filter_VolumeConfirm"
timeframe = "6h"
leverage = 1.0