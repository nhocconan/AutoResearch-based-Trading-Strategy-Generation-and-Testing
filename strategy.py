#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm_v2
Hypothesis: 6h Ichimoku TK cross with cloud filter from 1w trend and volume confirmation.
Long when TK crosses above cloud in bullish 1w trend (price > Senkou Span B_1w) with volume > 1.5x MA20.
Short when TK crosses below cloud in bearish 1w trend (price < Senkou Span B_1w) with volume spike.
Uses discrete sizing (0.25) and ATR(14) stop (2.0x) to manage drawdown.
Designed to catch strong trends while avoiding whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === Ichimoku components on 6h chart ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # Current cloud: Senkou A and B from 26 periods ago (already aligned for look-ahead safety)
    senkou_a_current = np.roll(senkou_a, 26)
    senkou_b_current = np.roll(senkou_b, 26)
    # Fill first 26 values with NaN
    senkou_a_current[:26] = np.nan
    senkou_b_current[:26] = np.nan
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_current, senkou_b_current)
    cloud_bottom = np.minimum(senkou_a_current, senkou_b_current)
    
    # TK Cross: Tenkan crossing above/below Kijun
    tk_cross_above = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_below = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # === 1w trend filter: price relative to Senkou Span B_1w ===
    df_1w_close = df_1w['close'].values
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    
    # Calculate 1w Ichimoku Senkou Span B
    period52_high_1w = pd.Series(df_1w_high).rolling(window=52, min_periods=52).max().values
    period52_low_1w = pd.Series(df_1w_low).rolling(window=52, min_periods=52).min().values
    senkou_b_1w = ((period52_high_1w + period52_low_1w) / 2)
    # Shift 26 periods ahead for cloud
    senkou_b_1w_shifted = np.roll(senkou_b_1w, 26)
    senkou_b_1w_shifted[:26] = np.nan
    
    # Align 1w Senkou Span B to 6h timeframe
    senkou_b_1w_aligned = align_htf_to_ltf(prices, df_1w, senkou_b_1w_shifted)
    
    # Bullish 1w trend: price above Senkou B_1w
    # Bearish 1w trend: price below Senkou B_1w
    bullish_1w = close > senkou_b_1w_aligned
    bearish_1w = close < senkou_b_1w_aligned
    
    # === Volume confirmation ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]) or
            np.isnan(senkou_b_1w_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        bullish = bullish_1w[i]
        bearish = bearish_1w[i]
        tk_above = tk_cross_above[i]
        tk_below = tk_cross_below[i]
        
        if position == 0:
            # Long: TK cross above + price above cloud + bullish 1w trend + volume spike
            long_condition = tk_above and (price > cloud_top[i]) and bullish and vol_spike
            # Short: TK cross below + price below cloud + bearish 1w trend + volume spike
            short_condition = tk_below and (price < cloud_bottom[i]) and bearish and vol_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit conditions: stoploss, TK cross below, or price below cloud
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            elif tk_cross_below[i] or price < cloud_bottom[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: stoploss, TK cross above, or price above cloud
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            elif tk_cross_above[i] or price > cloud_top[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_1wTrend_VolumeConfirm_v2"
timeframe = "6h"
leverage = 1.0