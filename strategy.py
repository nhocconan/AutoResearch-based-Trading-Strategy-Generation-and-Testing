#!/usr/bin/env python3
"""
6h_Ichimoku_KumoTwist_1dTrend_WeeklyVolume_v1
Hypothesis: Ichimoku cloud twist (Senkou Span A/B cross) on 6h timeframe, filtered by 1d trend (price vs EMA50) and weekly volume confirmation (>1.5x 4-week average), captures strong trend continuations while avoiding whipsaws. Works in bull/bear by only taking trades aligned with higher timeframe trend. Discrete sizing (0.25) targets 12-37 trades/year. ATR-based stoploss controls drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 4:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Weekly volume average (4-period)
    vol_1w = df_1w['volume'].values
    vol_ma_4w = pd.Series(vol_1w).rolling(window=4, min_periods=4).mean().values
    vol_ma_4w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_4w, additional_delay_bars=0)
    weekly_volume_confirm = volume > (vol_ma_4w_aligned * 1.5)
    
    # Ichimoku components (9, 26, 52 periods)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = ((tenkan + kijun) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 52 periods ahead
    period52_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of Ichimoku (52), EMA50 (50), ATR (14)
    start_idx = max(52, 50, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        trend_val = ema50_1d_aligned[i]
        atr_val = atr[i]
        vol_conf = weekly_volume_confirm[i]
        senkou_a_val = senkou_a[i]
        senkou_b_val = senkou_b[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val) or 
            np.isnan(senkou_a_val) or np.isnan(senkou_b_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Kumo twist detection: Senkou A crosses Senkou B
        # Bullish twist: Senkou A crosses above Senkou B (previous A <= previous B and current A > current B)
        # Bearish twist: Senkou A crosses below Senkou B (previous A >= previous B and current A < current B)
        if i >= 1:
            prev_senkou_a = senkou_a[i-1]
            prev_senkou_b = senkou_b[i-1]
            bullish_twist = (prev_senkou_a <= prev_senkou_b) and (senkou_a_val > senkou_b_val)
            bearish_twist = (prev_senkou_a >= prev_senkou_b) and (senkou_a_val < senkou_b_val)
        else:
            bullish_twist = False
            bearish_twist = False
        
        # Trend filter: price > 1d EMA50 = uptrend, price < 1d EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Cloud filter: price above cloud (bullish) or below cloud (bearish)
        # Cloud top is max(Senkou A, Senkou B), cloud bottom is min(Senkou A, Senkou B)
        cloud_top = max(senkou_a_val, senkou_b_val)
        cloud_bottom = min(senkou_a_val, senkou_b_val)
        price_above_cloud = close_val > cloud_top
        price_below_cloud = close_val < cloud_bottom
        
        # Entry conditions: Kumo twist in direction of 1d trend + price outside cloud + weekly volume
        long_entry = bullish_twist and is_uptrend and price_above_cloud and vol_conf
        short_entry = bearish_twist and is_downtrend and price_below_cloud and vol_conf
        
        # Exit conditions: ATR-based stoploss or opposite Kumo twist
        long_exit = False
        short_exit = False
        if position == 1:
            # Long stoploss: entry price - 2.0 * ATR
            stop_price = entry_price - 2.0 * atr_val
            long_exit = close_val < stop_price or bearish_twist  # Stop or bearish twist
        elif position == -1:
            # Short stoploss: entry price + 2.0 * ATR
            stop_price = entry_price + 2.0 * atr_val
            short_exit = close_val > stop_price or bullish_twist  # Stop or bullish twist
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Approximate entry price for stop calculation
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Ichimoku_KumoTwist_1dTrend_WeeklyVolume_v1"
timeframe = "6h"
leverage = 1.0