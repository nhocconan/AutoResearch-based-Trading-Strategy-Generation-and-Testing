#!/usr/bin/env python3
"""
Experiment #007: 6h Ichimoku Cloud Break + 1d Cloud Trend Filter

HYPOTHESIS: Ichimoku's Tenkan-Kijun cross on 6h captures medium-term momentum shifts,
while the 1d Ichimoku cloud (Senkou Span A/B) acts as a structural trend filter.
The cloud filter ensures we only trade in direction of the larger trend, avoiding
whipsaws in 2022's choppy bear market.

WHY IT WORKS IN BOTH MARKETS:
- Bull: TK cross above cloud = strong long signal
- Bear: TK cross below cloud = strong short signal  
- Range: Cloud thickness provides natural stop-loss buffer

WHY 6h: Between 4h and 12h in granularity. Cloud components (Tenkan=9, Kijun=26)
create meaningful crosses every 1-3 weeks per symbol. Should yield 75-150 total trades.

WHY ICHIMOKU: Not tried in any recent experiment. Uses lagging indicators (Chikou)
and leading indicators (Cloud) in combination for confluence.

CONFLUENCE:
1. TK-KJ cross on 6h (momentum)
2. Price vs 1d cloud (trend)
3. Volume spike confirmation (institutional)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_cloud_vol_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou_b=52):
    """Calculate Ichimoku Cloud components"""
    n = len(close)
    
    # Tenkan-sen (Conversion Line) = (9-period high + 9-period low) / 2
    tenkan_sen = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - tenkan + 1)
        tenkan_sen[i] = (np.max(high[start_idx:i+1]) + np.min(low[start_idx:i+1])) / 2
    
    # Kijun-sen (Base Line) = (26-period high + 26-period low) / 2
    kijun_sen = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - kijun + 1)
        kijun_sen[i] = (np.max(high[start_idx:i+1]) + np.min(low[start_idx:i+1])) / 2
    
    # Senkou Span A (Leading Span A) = (Tenkan + Kijun) / 2, plotted 26 periods ahead
    senkou_a = np.zeros(n)
    for i in range(n):
        senkou_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2
    
    # Senkou Span B (Leading Span B) = (52-period high + 52-period low) / 2
    senkou_b_val = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - senkou_b + 1)
        senkou_b_val[i] = (np.max(high[start_idx:i+1]) + np.min(low[start_idx:i+1])) / 2
    
    # Chikou Span (Lagging Span) = close, plotted 26 periods behind
    chikou_span = close.copy()
    
    return tenkan_sen, kijun_sen, senkou_a, senkou_b_val, chikou_span

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Ichimoku for cloud trend
    tenkan_1d, kijun_1d, span_a_1d, span_b_1d, chikou_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # 1d cloud: bullish if price above cloud, cloud top = max(span_a, span_b)
    # Need to shift cloud 26 periods back (Chikou concept) - use aligned version
    span_a_1d_aligned = align_htf_to_ltf(prices, df_1d, span_a_1d)
    span_b_1d_aligned = align_htf_to_ltf(prices, df_1d, span_b_1d)
    
    # === Local 6h Ichimoku ===
    tenkan_6h, kijun_6h, _, _, _ = calculate_ichimoku(high, low, close)
    
    # ATR for stops
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Previous TK cross state
    prev_tenkan = np.zeros(n)
    prev_kijun = np.zeros(n)
    prev_tenkan[0] = tenkan_6h[0]
    prev_kijun[0] = kijun_6h[0]
    for i in range(1, n):
        prev_tenkan[i] = tenkan_6h[i - 1]
        prev_kijun[i] = kijun_6h[i - 1]
    
    # Detect crosses
    tk_bull_cross = (prev_tenkan <= prev_kijun) & (tenkan_6h > kijun_6h)  # TK crossed above KJ
    tk_bear_cross = (prev_tenkan >= prev_kijun) & (tenkan_6h < kijun_6h)  # TK crossed below KJ
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    # Warmup: need 52 periods for Senkou B + 26 for cloud shift
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if cloud not aligned
        cloud_top = max(span_a_1d_aligned[i] if not np.isnan(span_a_1d_aligned[i]) else 0,
                       span_b_1d_aligned[i] if not np.isnan(span_b_1d_aligned[i]) else 0)
        if cloud_top == 0 or np.isnan(cloud_top):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === CLOUD FILTER (1d) ===
        # Bullish cloud: price above cloud top
        cloud_bullish = close[i] > cloud_top
        # Bearish cloud: price below cloud bottom (use min of spans)
        cloud_bottom = min(span_a_1d_aligned[i] if not np.isnan(span_a_1d_aligned[i]) else float('inf'),
                          span_b_1d_aligned[i] if not np.isnan(span_b_1d_aligned[i]) else float('inf'))
        cloud_bearish = close[i] < cloud_bottom
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TK crosses above KJ + price above cloud + volume ===
            if tk_bull_cross[i] and cloud_bullish and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: TK crosses below KJ + price below cloud + volume ===
            if tk_bear_cross[i] and cloud_bearish and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === HOLD PERIOD (minimum 3 bars = 18h to avoid churn) ===
        bars_held = i - entry_bar
        
        # === EXIT: TK re-cross ===
        if in_position and bars_held >= 3:
            # Exit long on TK bear cross
            if position_side > 0 and tk_bear_cross[i]:
                desired_signal = 0.0
            # Exit short on TK bull cross  
            if position_side < 0 and tk_bull_cross[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = close[i] - 2.5 * entry_atr
                else:
                    stop_price = close[i] + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals