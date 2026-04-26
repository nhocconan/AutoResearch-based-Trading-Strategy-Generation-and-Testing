#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm
Hypothesis: Ichimoku cloud breakout on 6h with 1d trend filter (price >/< Kumo twist) and volume confirmation (>1.5x average volume). 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year) on 6h timeframe.
Ichimoku provides dynamic support/resistance via Kumo cloud, TK cross for momentum, and works in both bull/bear markets via 1d trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need warmup for Ichimoku calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    # Not used for signals as it requires future data
    
    # Kumo (Cloud) boundaries: Senkou Span A and B
    # Future Kumo (for trend filter): shift Senkou spans back 26 periods to align with current price
    # We'll use the Kumo from 26 periods ago for current price comparison
    kumo_top = np.maximum(senkou_a, senkou_b)
    kumo_bottom = np.minimum(senkou_a, senkou_b)
    
    # TK Cross: Tenkan-sen crossing Kijun-sen
    tk_cross_up = (tenkan_sen > kijun_sen) & (np.roll(tenkan_sen, 1) <= np.roll(kijun_sen, 1))
    tk_cross_down = (tenkan_sen < kijun_sen) & (np.roll(tenkan_sen, 1) >= np.roll(kijun_sen, 1))
    
    # 1d trend filter: price relative to 1d Kumo
    df_1d_close = df_1d['close'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    
    # Calculate 1d Ichimoku for trend filter
    period9_high_1d = pd.Series(df_1d_high).rolling(window=9, min_periods=9).max().values
    period9_low_1d = pd.Series(df_1d_low).rolling(window=9, min_periods=9).min().values
    tenkan_sen_1d = (period9_high_1d + period9_low_1d) / 2
    
    period26_high_1d = pd.Series(df_1d_high).rolling(window=26, min_periods=26).max().values
    period26_low_1d = pd.Series(df_1d_low).rolling(window=26, min_periods=26).min().values
    kijun_sen_1d = (period26_high_1d + period26_low_1d) / 2
    
    senkou_a_1d = ((tenkan_sen_1d + kijun_sen_1d) / 2)
    period52_high_1d = pd.Series(df_1d_high).rolling(window=52, min_periods=52).max().values
    period52_low_1d = pd.Series(df_1d_low).rolling(window=52, min_periods=52).min().values
    senkou_b_1d = ((period52_high_1d + period52_low_1d) / 2)
    
    kumo_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Align 1d Kumo to 6h timeframe (price above/below 1d cloud indicates trend)
    kumo_top_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d)
    kumo_bottom_1d_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Start after warmup (need 52 for Senkou B)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Get current values
        tenkan = tenkan_sen[i]
        kijun = kijun_sen[i]
        close_val = close[i]
        kumo_top_now = kumo_top[i]
        kumo_bottom_now = kumo_bottom[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        kumo_top_1d = kumo_top_1d_aligned[i]
        kumo_bottom_1d = kumo_bottom_1d_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(tenkan) or np.isnan(kijun) or np.isnan(kumo_top_now) or 
            np.isnan(kumo_bottom_now) or np.isnan(avg_vol) or 
            np.isnan(kumo_top_1d) or np.isnan(kumo_bottom_1d)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Kumo twist: Senkou A crossing Senkou B (trend change indicator)
        kumo_twist_up = (senkou_a[i] > senkou_b[i]) & (senkou_a[i-1] <= senkou_b[i-1])
        kumo_twist_down = (senkou_a[i] < senkou_b[i]) & (senkou_a[i-1] >= senkou_b[i-1])
        
        # Long conditions:
        # 1. Price breaks above Kumo (bullish breakout)
        # 2. TK cross bullish (Tenkan crosses above Kijun)
        # 3. 1d trend filter: price above 1d Kumo (bullish long-term trend)
        # 4. Volume confirmation
        long_condition = (
            (close_val > kumo_top_now) and  # Price above cloud
            tk_cross_up[i] and              # Bullish TK cross
            (close_val > kumo_top_1d) and   # Price above 1d cloud (bullish trend)
            volume_confirmed
        )
        
        # Short conditions:
        # 1. Price breaks below Kumo (bearish breakout)
        # 2. TK cross bearish (Tenkan crosses below Kijun)
        # 3. 1d trend filter: price below 1d Kumo (bearish long-term trend)
        # 4. Volume confirmation
        short_condition = (
            (close_val < kumo_bottom_now) and  # Price below cloud
            tk_cross_down[i] and               # Bearish TK cross
            (close_val < kumo_bottom_1d) and   # Price below 1d cloud (bearish trend)
            volume_confirmed
        )
        
        # Exit conditions: TK cross in opposite direction or price re-enters Kumo
        exit_long = tk_cross_down[i] or (close_val < kumo_top_now and close_val > kumo_bottom_now)
        exit_short = tk_cross_up[i] or (close_val < kumo_top_now and close_val > kumo_bottom_now)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0