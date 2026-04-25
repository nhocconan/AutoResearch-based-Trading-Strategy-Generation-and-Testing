#!/usr/bin/env python3
"""
6h Ichimoku Cloud + 1d Trend Filter + Volume Spike
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. TK cross + cloud filter confirms momentum direction. 1d EMA50 trend filter avoids counter-trend trades. Volume spike ensures participation. Works in bull via buying above cloud with TK cross up, bear via selling below cloud with TK cross down. Discrete sizing (0.25) controls drawdown. Target: 12-37 trades/year on 6h.
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
    
    # Get 1d data for EMA50 trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute ATR(14) for stoploss
    atr = np.full(n, 0.0)
    if n >= 14:
        tr1 = np.abs(np.diff(high, prepend=high[0]))
        tr2 = np.abs(np.diff(close, prepend=close[0]))
        tr3 = np.abs(np.diff(low, prepend=low[0]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 20-period volume MA for volume spike detection
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    # Ichimoku components (9, 26, 52)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = np.zeros(n)
    period9_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 8)
        period9_high[i] = np.max(high[start_idx:i+1])
        period9_low[i] = np.min(low[start_idx:i+1])
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = np.zeros(n)
    period26_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 25)
        period26_high[i] = np.max(high[start_idx:i+1])
        period26_low[i] = np.min(low[start_idx:i+1])
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = np.zeros(n)
    period52_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 51)
        period52_high[i] = np.max(high[start_idx:i+1])
        period52_low[i] = np.min(low[start_idx:i+1])
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud: Senkou Span A and B shifted back 26 periods (to align with current price)
    senkou_a_lag = np.full(n, np.nan)
    senkou_b_lag = np.full(n, np.nan)
    for i in range(26, n):
        senkou_a_lag[i] = senkou_a[i - 26]
        senkou_b_lag[i] = senkou_b[i - 26]
    
    # Cloud top/bottom
    cloud_top = np.maximum(senkou_a_lag, senkou_b_lag)
    cloud_bottom = np.minimum(senkou_a_lag, senkou_b_lag)
    
    # TK Cross: Tenkan-sen crossing above/below Kijun-sen
    tk_cross_up = np.zeros(n, dtype=bool)
    tk_cross_down = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or np.isnan(tenkan_sen[i-1]) or np.isnan(kijun_sen[i-1])):
            tk_cross_up[i] = (tenkan_sen[i] > kijun_sen[i]) and (tenkan_sen[i-1] <= kijun_sen[i-1])
            tk_cross_down[i] = (tenkan_sen[i] < kijun_sen[i]) and (tenkan_sen[i-1] >= kijun_sen[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Ichimoku (52) and EMA50 to propagate
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i]) or
            np.isnan(cloud_top[i]) or
            np.isnan(cloud_bottom[i]) or
            np.isnan(tenkan_sen[i]) or
            np.isnan(kijun_sen[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            # Long: price above cloud AND TK cross up AND uptrend (price > 1d EMA50) AND volume spike
            long_condition = (curr_close > cloud_top[i]) and tk_cross_up[i] and (curr_close > ema_50) and volume_spike
            # Short: price below cloud AND TK cross down AND downtrend (price < 1d EMA50) AND volume spike
            short_condition = (curr_close < cloud_bottom[i]) and tk_cross_down[i] and (curr_close < ema_50) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: stoploss (2.0*ATR below entry) or price falls below cloud or TK cross down
            if curr_close <= entry_price - 2.0 * atr_val or curr_close < cloud_bottom[i] or tk_cross_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: stoploss (2.0*ATR above entry) or price rises above cloud or TK cross up
            if curr_close >= entry_price + 2.0 * atr_val or curr_close > cloud_top[i] or tk_cross_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_TKCross_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0