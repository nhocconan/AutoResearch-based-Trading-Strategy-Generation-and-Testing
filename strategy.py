#!/usr/bin/env python3
"""
Hypothesis: 6h Ichimoku Cloud strategy with 1d EMA trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter (bull/bear regime) and volume spike detection.
- Ichimoku Components:
  * Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
  * Kijun-sen (Base Line): (26-period high + 26-period low)/2
  * Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 plotted 26 periods ahead
  * Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
  * Kumo (Cloud): Area between Senkou Span A and B
- Entry: Long when price > Kumo AND Tenkan > Kijun AND 1d EMA34 up AND volume > 1.5 * 20-period average volume.
         Short when price < Kumo AND Tenkan < Kijun AND 1d EMA34 down AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Ichimoku signal (price crosses Kumo in opposite direction OR Tenkan/Kijun cross reverses).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets via cloud breakouts with trend alignment, avoids false signals in ranging markets via volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:  # Need sufficient data for Ichimoku calculations (52+26)
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
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Ichimoku components on 6h data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period_tenkan = 9
    tenkan_sen = (pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max() + 
                  pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period_kijun = 26
    kijun_sen = (pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max() + 
                 pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period_senkou_b = 52
    senkou_span_b = (pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max() + 
                     pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min()) / 2
    
    # The Ichimoku lines are plotted ahead, but for current cloud we use current values
    # The cloud (Kumo) is between Senkou Span A and Senkou Span B
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    # Need max of: 52 (Senkou B), 26 (Kijun), 9 (Tenkan), 34 (1d EMA), 20 (1d volume MA)
    start_idx = max(52, 26, 9, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(tenkan_sen.iloc[i]) or np.isnan(kijun_sen.iloc[i]) or
            np.isnan(senkou_span_a.iloc[i]) or np.isnan(senkou_span_b.iloc[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Current Ichimoku values
        tenkan = tenkan_sen.iloc[i]
        kijun = kijun_sen.iloc[i]
        span_a = senkou_span_a.iloc[i]
        span_b = senkou_span_b.iloc[i]
        
        # Kumo (Cloud) boundaries
        upper_cloud = max(span_a, span_b)
        lower_cloud = min(span_a, span_b)
        
        # Trend filter from 1d EMA34: comparing current to previous value
        ema34_now = ema34_1d_aligned[i]
        ema34_prev = ema34_1d_aligned[i-1] if i > 0 else ema34_now
        ema34_up = ema34_now > ema34_prev
        ema34_down = ema34_now < ema34_prev
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Ichimoku signals
        price_above_cloud = curr_close > upper_cloud
        price_below_cloud = curr_close < lower_cloud
        tenkan_above_kijun = tenkan > kijun
        tenkan_below_kijun = tenkan < kijun
        
        # Exit conditions: opposite Ichimoku signal
        if position != 0:
            # Exit long: price falls below cloud OR Tenkan crosses below Kijun
            if position == 1:
                if (price_below_cloud or (tenkan_below_kijun and tenkan_sen.iloc[i-1] >= kijun_sen.iloc[i-1])):
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price rises above cloud OR Tenkan crosses above Kijun
            elif position == -1:
                if (price_above_cloud or (tenkan_above_kijun and tenkan_sen.iloc[i-1] <= kijun_sen.iloc[i-1])):
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Ichimoku breakout with trend and volume filters
        if position == 0:
            # Long: price > cloud AND Tenkan > Kijun AND 1d EMA34 up AND volume confirmation
            long_condition = (price_above_cloud and 
                            tenkan_above_kijun and
                            ema34_up and
                            volume_confirm)
            
            # Short: price < cloud AND Tenkan < Kijun AND 1d EMA34 down AND volume confirmation
            short_condition = (price_below_cloud and 
                             tenkan_below_kijun and
                             ema34_down and
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