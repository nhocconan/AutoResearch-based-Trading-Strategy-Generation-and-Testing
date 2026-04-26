#!/usr/bin/env python3
"""
6h_Ichimoku_Kumo_Twist_1wTrend_HTFVolSpike_v1
Hypothesis: Use 6h timeframe with Ichimoku cloud twist (Tenkan/Kijun cross) confirmed by 1w EMA50 trend and 1d volume spike. Targets 12-30 trades/year to minimize fee drag. Works in bull/bear markets by using 1w EMA50 for trend direction and volume confirmation to filter false signals. Includes ATR-based stoploss to manage risk.
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
    
    # Calculate Ichimoku components on 6h
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2
    senkou_a = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = (period52_high + period52_low) / 2
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 52 for Senkou B, 50 for 1w EMA, 20 for volume avg, 14 for ATR
    start_idx = max(52, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr[i]
        size = 0.25  # 25% position size
        
        # Determine cloud direction (bullish if Senkou A > Senkou B)
        cloud_bullish = senkou_a[i] > senkou_b[i]
        
        if position == 0:
            # Flat - look for TK cross with trend and volume confirmation
            # Long: Tenkan crosses above Kijun + 1w EMA50 uptrend + volume spike + price above cloud
            tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            long_entry = tk_cross_up and \
                       (ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]) and \
                       volume_spike[i] and \
                       (close_val > senkou_a[i] and close_val > senkou_b[i])
            
            # Short: Tenkan crosses below Kijun + 1w EMA50 downtrend + volume spike + price below cloud
            tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            short_entry = tk_cross_down and \
                        (ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]) and \
                        volume_spike[i] and \
                        (close_val < senkou_a[i] and close_val < senkou_b[i])
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on TK cross down or ATR stoploss
            tk_cross_down = tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]
            exit_condition = tk_cross_down or \
                           (close_val < entry_price - 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on TK cross up or ATR stoploss
            tk_cross_up = tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]
            exit_condition = tk_cross_up or \
                           (close_val > entry_price + 2.5 * atr_val)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Kumo_Twist_1wTrend_HTFVolSpike_v1"
timeframe = "6h"
leverage = 1.0