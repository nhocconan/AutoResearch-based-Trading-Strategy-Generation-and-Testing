#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter (price > EMA34 = bull trend, price < EMA34 = bear trend).
- Ichimoku Components (6h):
  * Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
  * Kijun-sen (Base Line): (26-period high + 26-period low) / 2
  * Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
  * Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
  * Kumo (Cloud): between Senkou Span A and Senkou Span B
- Entry: Long when price > Cloud AND Tenkan > Kijun AND 1d bull trend AND volume > 1.5 * 20-period average volume.
         Short when price < Cloud AND Tenkan < Kijun AND 1d bear trend AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Tenkan/Kijun cross (Tenkan < Kijun for long exit, Tenkan > Kijun for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets via long cloud breaks and in bear markets via short cloud breaks, with 1d trend filter preventing counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 6h Ichimoku components
    # Tenkan-sen (Conversion Line): 9-period high/low
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen (Base Line): 26-period high/low
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2.0
    
    # Senkou Span B (Leading Span B): 52-period high/low
    high_52 = pd.Series(high).rolling(window=52, min_periods=52).max().values
    low_52 = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (high_52 + low_52) / 2.0
    
    # Calculate 6h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 34)  # Need 52 for Senkou B, 34 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Determine cloud boundaries (Senkou Span A/B)
        upper_cloud = max(senkou_a[i], senkou_b[i])
        lower_cloud = min(senkou_a[i], senkou_b[i])
        
        # 1d trend filter: price > EMA34 = bull trend, price < EMA34 = bear trend
        bull_trend = close[i] > ema34_1d_aligned[i]
        bear_trend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i] if not np.isnan(vol_ma_20[i]) else False
        
        # Ichimoku signals: Tenkan/Kijun cross
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        
        # Exit conditions: opposite Tenkan/Kijun cross
        if position != 0:
            # Exit long: Tenkan < Kijun
            if position == 1:
                if tenkan_below_kijun:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Tenkan > Kijun
            elif position == -1:
                if tenkan_above_kijun:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Cloud breakout with trend and volume filters
        if position == 0:
            # Long: price > Upper Cloud AND Tenkan > Kijun AND bull trend AND volume confirmation
            long_condition = (curr_close > upper_cloud and 
                            tenkan_above_kijun and
                            bull_trend and
                            volume_confirm)
            
            # Short: price < Lower Cloud AND Tenkan < Kijun AND bear trend AND volume confirmation
            short_condition = (curr_close < lower_cloud and 
                             tenkan_below_kijun and
                             bear_trend and
                             volume_confirm)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_1dEMA34Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0