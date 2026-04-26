#!/usr/bin/env python3
"""
6h_Ichimoku_Cloud_Breakout_WeeklyTrend
Hypothesis: 6h Ichimoku cloud breakout with weekly trend filter. Long when price breaks above cloud with bullish TK cross and weekly trend up; short when price breaks below cloud with bearish TK cross and weekly trend down. Uses volume confirmation to avoid false breakouts. Ichimoku provides dynamic support/resistance and trend direction, working in both bull and bear markets via weekly trend filter.
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
    
    # Load weekly data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Ichimoku components (9, 26, 52 periods)
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
    max_high_senkou = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_b = ((max_high_senkou + min_low_senkou) / 2)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind (not used for signals)
    
    # The cloud is between Senkou Span A and Senkou Span B
    # For signal generation, we use current cloud (Senkou A/B from 26 periods ago)
    # So we need to shift Senkou A/B back by 26 periods to align with current price
    cloud_top = np.maximum(senkou_a, senkou_b)
    cloud_bottom = np.minimum(senkou_a, senkou_b)
    
    # Shift cloud components back by 26 periods to align with current price
    cloud_top_aligned = np.roll(cloud_top, 26)
    cloud_bottom_aligned = np.roll(cloud_bottom, 26)
    # Set first 26 values to NaN since they don't have valid cloud data
    cloud_top_aligned[:26] = np.nan
    cloud_bottom_aligned[:26] = np.nan
    
    # Calculate ATR(14) for volume confirmation
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period ATR) for volume regime
    atr_ratio = atr / pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # TK Cross (Tenkan/Kijun cross)
    tk_cross = tenkan - kijun  # >0 bullish, <0 bearish
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (52 for Senkou B, 26 for TK cross/cloud shift, 50 for ATR ratio)
    start_idx = 52
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or
            np.isnan(cloud_top_aligned[i]) or np.isnan(cloud_bottom_aligned[i]) or
            np.isnan(tk_cross[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        tk_val = tk_cross[i]
        cloud_top_val = cloud_top_aligned[i]
        cloud_bottom_val = cloud_bottom_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = atr_ratio[i] > 1.2  # volume confirmation
        size = fixed_size
        
        # Determine cloud relationship
        price_above_cloud = close_val > cloud_top_val
        price_below_cloud = close_val < cloud_bottom_val
        price_in_cloud = (close_val >= cloud_bottom_val) and (close_val <= cloud_top_val)
        
        # Entry conditions: Ichimoku breakout with TK cross alignment AND weekly trend AND volume
        long_entry = price_above_cloud and (tk_val > 0) and (close_val > ema_50_val) and vol_spike
        short_entry = price_below_cloud and (tk_val < 0) and (close_val < ema_50_val) and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when price re-enters cloud or TK cross turns bearish
            if price_in_cloud or (tk_val < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price re-enters cloud or TK cross turns bullish
            if price_in_cloud or (tk_val > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Ichimoku_Cloud_Breakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0