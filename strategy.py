#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend filter and volume spike confirmation.
- Ichimoku Components:
  * Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
  * Kijun-sen (Base Line): (26-period high + 26-period low)/2
  * Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
  * Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
  * Kumo (Cloud): between Senkou Span A and B
- Entry: Long when price > Kumo AND Tenkan > Kijun AND 1d EMA50 uptrend AND volume > 1.5 * 20-period average volume.
         Short when price < Kumo AND Tenkan < Kijun AND 1d EMA50 downtrend AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Ichimoku signal (price crosses Kumo in opposite direction).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by only trading with 1d trend filter, avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:  # Need sufficient data for Ichimoku calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): 9-period high/low
    period_tenkan = 9
    highest_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    lowest_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan = (highest_tenkan + lowest_tenkan) / 2
    
    # Kijun-sen (Base Line): 26-period high/low
    period_kijun = 26
    highest_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    lowest_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun = (highest_kijun + lowest_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): 52-period high/low
    period_senkou_b = 52
    highest_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    lowest_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = (highest_senkou_b + lowest_senkou_b) / 2
    
    # For cloud calculation, we need to shift Senkou spans forward by 26 periods
    # But for signal generation at time t, we use Senkou values that were calculated 26 periods ago
    # So we compare current price with Senkou values from 26 periods back
    senkou_a_lagged = np.roll(senkou_a, 26)
    senkou_b_lagged = np.roll(senkou_b, 26)
    # First 26 values will be invalid due to roll, set to nan
    senkou_a_lagged[:26] = np.nan
    senkou_b_lagged[:26] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(52, 26) + 26  # Need 52 for Senkou B, plus 26 for lag
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(senkou_a_lagged[i]) or np.isnan(senkou_b_lagged[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 slope (using current vs 5 periods ago)
        if i >= 5:
            ema50_now = ema50_1d_aligned[i]
            ema50_prev = ema50_1d_aligned[i-5]
            ema50_uptrend = ema50_now > ema50_prev
            ema50_downtrend = ema50_now < ema50_prev
        else:
            ema50_uptrend = False
            ema50_downtrend = False
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Kumo (Cloud) boundaries: Senkou Span A and B
        top_cloud = max(senkou_a_lagged[i], senkou_b_lagged[i])
        bottom_cloud = min(senkou_a_lagged[i], senkou_b_lagged[i])
        
        # Ichimoku signals
        tenkan_above_kijun = tenkan[i] > kijun[i]
        tenkan_below_kijun = tenkan[i] < kijun[i]
        price_above_cloud = curr_close > top_cloud
        price_below_cloud = curr_close < bottom_cloud
        
        # Exit conditions: opposite Ichimoku signal
        if position != 0:
            # Exit long: price < Cloud OR Tenkan < Kijun
            if position == 1:
                if price_below_cloud or tenkan_below_kijun:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price > Cloud OR Tenkan > Kijun
            elif position == -1:
                if price_above_cloud or tenkan_above_kijun:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Ichimoku signal with trend and volume filters
        if position == 0:
            # Long: price > Cloud AND Tenkan > Kijun AND 1d EMA50 uptrend AND volume confirmation
            long_condition = (price_above_cloud and 
                            tenkan_above_kijun and
                            ema50_uptrend and
                            volume_confirm)
            
            # Short: price < Cloud AND Tenkan < Kijun AND 1d EMA50 downtrend AND volume confirmation
            short_condition = (price_below_cloud and 
                             tenkan_below_kijun and
                             ema50_downtrend and
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

name = "6h_Ichimoku_Cloud_1dEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0