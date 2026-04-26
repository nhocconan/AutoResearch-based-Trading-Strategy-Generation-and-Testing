#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout on 12h with 1d trend filter (price > 1d EMA50 = long, price < 1d EMA50 = short) and volume confirmation (>2.0x 20-bar average) captures strong trending moves with fewer whipsaws. Uses discrete sizing (0.25) and ATR-based stoploss (2.0x ATR) to target 12-30 trades/year. Works in bull/bear by only taking breakouts aligned with 1d trend. Uses 12h primary timeframe as instructed.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # ATR(14) for stoploss calculation
    tr1 = pd.Series(high[1:] - low[1:]).values
    tr2 = pd.Series(np.abs(high[1:] - close[:-1])).values
    tr3 = pd.Series(np.abs(low[1:] - close[:-1])).values
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Camarilla pivot levels from previous 12h bar (using typical price)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Calculate pivot based on previous bar's typical price
    prev_typical = pd.Series(typical_price).shift(1)
    prev_high = pd.Series(high).shift(1)
    prev_low = pd.Series(low).shift(1)
    pivot = (prev_high + prev_low + prev_typical) / 3.0
    # Camarilla levels
    range_val = prev_high - prev_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    
    # Warmup: max of EMA50 (50), ATR (14), volume MA (20), pivot calculation (need 2 bars for shift)
    start_idx = max(50, 14, 20, 2)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        trend_val = ema50_1d_aligned[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        r1_val = r1[i]
        s1_val = s1[i]
        pivot_val = pivot[i]
        
        # Skip if any data not ready
        if (np.isnan(trend_val) or np.isnan(atr_val) or np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(pivot_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: price > 1d EMA50 = uptrend, price < 1d EMA50 = downtrend
        is_uptrend = close_val > trend_val
        is_downtrend = close_val < trend_val
        
        # Camarilla breakout conditions
        long_breakout = close_val > r1_val  # Break above R1
        short_breakout = close_val < s1_val  # Break below S1
        
        # Entry conditions: Camarilla breakout in direction of 1d trend + volume
        long_entry = long_breakout and is_uptrend and vol_conf
        short_entry = short_breakout and is_downtrend and vol_conf
        
        # Exit conditions: ATR-based stoploss or opposite Camarilla level touch
        long_exit = False
        short_exit = False
        if position == 1:
            # Long stoploss: entry price - 2.0 * ATR
            stop_price = entry_price - 2.0 * atr_val
            long_exit = close_val < stop_price or close_val < s1_val  # Stop or touch S1
        elif position == -1:
            # Short stoploss: entry price + 2.0 * ATR
            stop_price = entry_price + 2.0 * atr_val
            short_exit = close_val > stop_price or close_val > r1_val  # Stop or touch R1
        
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0