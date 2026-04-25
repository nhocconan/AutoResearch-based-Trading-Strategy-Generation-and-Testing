#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike
Hypothesis: Ichimoku cloud acts as dynamic support/resistance on 6h timeframe.
Long when price > cloud with bullish TK cross, 1d EMA trend alignment, and volume spike.
Short when price < cloud with bearish TK cross, 1d EMA trend alignment, and volume spike.
Uses 1d EMA34 for higher timeframe trend filter to avoid counter-trend trades.
Targets 12-30 trades/year by requiring confluence of multiple filters.
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
    open_price = prices['open'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Ichimoku components (calculated on 6h data)
    # Conversion Line (Tenkan-sen): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Base Line (Kijun-sen): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Leading Span A (Senkou Span A): (Conversion Line + Base Line)/2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    
    # Leading Span B (Senkou Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_span_b = (period52_high + period52_low) / 2
    
    # The cloud is between Senkou Span A and B
    # Actual cloud plotted 26 periods ahead, but for current price we use current values
    # Upper cloud boundary = max(Senkou Span A, Senkou Span B)
    # Lower cloud boundary = min(Senkou Span A, Senkou Span B)
    upper_cloud = np.maximum(senkou_span_a, senkou_span_b)
    lower_cloud = np.minimum(senkou_span_a, senkou_span_b)
    
    # TK Cross: Tenkan-sen crossing Kijun-sen
    tk_cross_bullish = tenkan_sen > kijun_sen
    tk_cross_bearish = tenkan_sen < kijun_sen
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Ichimoku calculations (52) and 1d EMA (34)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(senkou_span_a[i]) or
            np.isnan(senkou_span_b[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with Ichimoku conditions, trend alignment, and volume
            # Long: price above cloud, bullish TK cross, uptrend, volume confirmation
            long_signal = (curr_close > upper_cloud[i]) and tk_cross_bullish[i] and uptrend and volume_confirm[i]
            # Short: price below cloud, bearish TK cross, downtrend, volume confirmation
            short_signal = (curr_close < lower_cloud[i]) and tk_cross_bearish[i] and downtrend and volume_confirm[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price falls below cloud (Senkou Span A) or trend changes
            if curr_close < senkou_span_a[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price rises above cloud (Senkou Span A) or trend changes
            if curr_close > senkou_span_a[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Filter_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0