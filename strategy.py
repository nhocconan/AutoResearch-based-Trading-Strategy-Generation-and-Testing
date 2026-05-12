#!/usr/bin/env python3
name = "6h_ADX_Ichimoku_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # ADX on 6h
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Ichimoku on 6h (conversion: 9, base: 26, span B: 52)
    nine_period_high = pd.Series(high).rolling(window=9, min_periods=9).max().values
    nine_period_low = pd.Series(low).rolling(window=9, min_periods=9).min().values
    conversion_line = (nine_period_high + nine_period_low) / 2
    
    period26_high = pd.Series(high).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low).rolling(window=26, min_periods=26).min().values
    base_line = (period26_high + period26_low) / 2
    
    span_a = (conversion_line + base_line) / 2
    span_b_high = pd.Series(high).rolling(window=52, min_periods=52).max().values
    span_b_low = pd.Series(low).rolling(window=52, min_periods=52).min().values
    span_b = (span_b_high + span_b_low) / 2
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 52)  # need enough data for 1w EMA50 and Ichimoku
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(adx[i]) or np.isnan(conversion_line[i]) or np.isnan(base_line[i]) or np.isnan(span_a[i]) or np.isnan(span_b[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Ichimoku cloud
        span_a_lagged = np.roll(span_a, 26)  # projected 26 periods ahead
        span_b_lagged = np.roll(span_b, 26)
        span_a_lagged[:26] = np.nan
        span_b_lagged[:26] = np.nan
        
        # Cloud top and bottom
        cloud_top = np.maximum(span_a_lagged, span_b_lagged)
        cloud_bottom = np.minimum(span_a_lagged, span_b_lagged)
        
        if position == 0:
            # Long: price above cloud, bullish TK cross, ADX > 25, 1w uptrend
            if (close[i] > cloud_top[i] and 
                conversion_line[i] > base_line[i] and 
                adx[i] > 25 and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below cloud, bearish TK cross, ADX > 25, 1w downtrend
            elif (close[i] < cloud_bottom[i] and 
                  conversion_line[i] < base_line[i] and 
                  adx[i] > 25 and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below cloud or TK cross turns bearish
            if (close[i] < cloud_bottom[i] or conversion_line[i] < base_line[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above cloud or TK cross turns bullish
            if (close[i] > cloud_top[i] or conversion_line[i] > base_line[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals