#!/usr/bin/env python3
"""
6h_Ichimoku_TK_Cross_12hCloud_Filter_VolumeSpike
Hypothesis: 6h Ichimoku Tenkan/Kijun cross with 12h cloud filter and volume confirmation.
Uses 12h for higher timeframe trend/cloud context to reduce whipsaw, 6h for precise entry timing.
Volume spike confirms momentum. Designed for 12-37 trades/year on BTC/ETH.
Works in bull/bear via cloud acting as dynamic support/resistance and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 12h data for Ichimoku cloud (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_12h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_12h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_12h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_12h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 plotted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 plotted 26 periods ahead
    period52_high = pd.Series(high_12h).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_12h).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Align 12h Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_12h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_12h, kijun_sen)
    senkou_a_aligned = align_htf_to_ltf(prices, df_12h, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_12h, senkou_b)
    
    # Cloud top/bottom (Senkou Span A/B)
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Ichimoku (52) and volume MA (20)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to cloud
        above_cloud = curr_close > cloud_top[i]
        below_cloud = curr_close < cloud_bottom[i]
        in_cloud = ~(above_cloud | below_cloud)
        
        if position == 0:
            # Look for entry signals with TK cross, cloud filter, and volume spike
            # Tenkan-sen crossing above Kijun-sen (bullish cross)
            tk_bullish_cross = (tenkan_aligned[i] > kijun_aligned[i]) and (tenkan_aligned[i-1] <= kijun_aligned[i-1])
            # Tenkan-sen crossing below Kijun-sen (bearish cross)
            tk_bearish_cross = (tenkan_aligned[i] < kijun_aligned[i]) and (tenkan_aligned[i-1] >= kijun_aligned[i-1])
            
            # Long: bullish TK cross + price above cloud + volume spike
            long_entry = tk_bullish_cross and above_cloud and volume_spike[i]
            # Short: bearish TK cross + price below cloud + volume spike
            short_entry = tk_bearish_cross and below_cloud and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Stoploss: 2.0 * ATR below entry (using 6h ATR)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if TK cross turns bearish or price falls below cloud
            elif tk_bearish_cross or curr_close < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Calculate 6h ATR (same as above)
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
            atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
            
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit if TK cross turns bullish or price rises above cloud
            elif tk_bullish_cross or curr_close > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_TK_Cross_12hCloud_Filter_VolumeSpike"
timeframe = "6h"
leverage = 1.0