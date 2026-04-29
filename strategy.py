#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with TK Cross and 1d trend filter
# Ichimoku: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (26/52-period)
# Long when TK Cross bullish AND price > Cloud AND 1d EMA50 uptrend AND volume > 1.5x 20-period average
# Short when TK Cross bearish AND price < Cloud AND 1d EMA50 downtrend AND volume > 1.5x 20-period average
# Uses 6h timeframe for lower trade frequency (~20-40 trades/year) and discrete sizing (0.25) to minimize fee drag
# Works in both bull and bear markets by combining momentum (TK Cross) with trend filter (1d EMA50) and cloud filter

name = "6h_Ichimoku_TK_Cross_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components (using 6h data)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    period_tenkan = 9
    max_high_tenkan = pd.Series(high).rolling(window=period_tenkan, min_periods=period_tenkan).max().values
    min_low_tenkan = pd.Series(low).rolling(window=period_tenkan, min_periods=period_tenkan).min().values
    tenkan_sen = (max_high_tenkan + min_low_tenkan) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    period_kijun = 26
    max_high_kijun = pd.Series(high).rolling(window=period_kijun, min_periods=period_kijun).max().values
    min_low_kijun = pd.Series(low).rolling(window=period_kijun, min_periods=period_kijun).min().values
    kijun_sen = (max_high_kijun + min_low_kijun) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2 shifted 26 periods ahead
    period_senkou_b = 52
    max_high_senkou_b = pd.Series(high).rolling(window=period_senkou_b, min_periods=period_senkou_b).max().values
    min_low_senkou_b = pd.Series(low).rolling(window=period_senkou_b, min_periods=period_senkou_b).min().values
    senkou_span_b = ((max_high_senkou_b + min_low_senkou_b) / 2)
    
    # Chikou Span (Lagging Span): Close shifted -26 periods behind (not used for signals)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 52, 26, 9, 14)  # warmup for indicators (need 52 for Senkou B)
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        curr_tenkan = tenkan_sen[i]
        curr_kijun = kijun_sen[i]
        # Cloud boundaries: Senkou Span A and B shifted forward by 26 periods
        # So we use values from 26 periods ago for current cloud
        if i >= 26:
            span_a = senkou_span_a[i-26]
            span_b = senkou_span_b[i-26]
            # Cloud top is max of spans, cloud bottom is min of spans
            cloud_top = max(span_a, span_b)
            cloud_bottom = min(span_a, span_b)
        else:
            # Not enough data for cloud, use current values (will be invalid but protected by start_idx)
            span_a = senkou_span_a[i]
            span_b = senkou_span_b[i]
            cloud_top = max(span_a, span_b)
            cloud_bottom = min(span_a, span_b)
        
        # TK Cross: Tenkan-sen crossing Kijun-sen
        bullish_tk_cross = curr_tenkan > curr_kijun and tenkan_sen[i-1] <= kijun_sen[i-1]
        bearish_tk_cross = curr_tenkan < curr_kijun and tenkan_sen[i-1] >= kijun_sen[i-1]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR TK Cross turns bearish
            if curr_close < stop_price or bearish_tk_cross:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop OR TK Cross turns bullish
            if curr_close > stop_price or bullish_tk_cross:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bullish TK Cross AND price > Cloud AND price > 1d EMA50 AND volume spike
            if (bullish_tk_cross and 
                curr_close > cloud_top and 
                curr_close > curr_ema_1d and 
                vol_spike):
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: Bearish TK Cross AND price < Cloud AND price < 1d EMA50 AND volume spike
            elif (bearish_tk_cross and 
                  curr_close < cloud_bottom and 
                  curr_close < curr_ema_1d and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals