#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_VolumeRegime_1wTrend
Hypothesis: On daily timeframe, use Camarilla H3/L3 levels from previous day with volume regime filter (ATR ratio > 1.0) and 1-week EMA50 trend filter. In uptrend (price > weekly EMA50): long at H3 breakout, short at S3 breakdown. In downtrend: mean reversion at H3/L3. Uses discrete sizing 0.25 to target ~15-25 trades/year. Works in bull/bear via trend-adaptive logic.
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
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous day
    close_prev = df_1d['close'].shift(1).values
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    range_prev = high_prev - low_prev
    
    camarilla_h3 = close_prev + range_prev * 1.1 / 2
    camarilla_l3 = close_prev - range_prev * 1.1 / 2
    camarilla_h4 = close_prev + range_prev * 1.1  # Stoploss reference
    camarilla_l4 = close_prev - range_prev * 1.1  # Stoploss reference
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(10) for volume regime
    atr_period = 10
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate ATR ratio (current ATR / 20-period ATR) for volume regime
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr / atr_ma
    
    # Align all HTF arrays to 1d
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)  # ATR ratio is 1d-based
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 20 for ATR ratio, 50 for weekly EMA, plus 1 for daily shift
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = atr_ratio_aligned[i] > 1.0
        size = fixed_size
        
        # Determine trend: price > weekly EMA50 = uptrend
        is_uptrend = close_val > ema_50_val
        
        if position == 0:
            # Flat - look for entry
            if is_uptrend:
                # Uptrend: breakout strategy
                long_entry = (close_val > h3) and vol_spike
                short_entry = (close_val < l3) and vol_spike
            else:
                # Downtrend: mean reversion strategy
                long_entry = (close_val < l3) and vol_spike
                short_entry = (close_val > h3) and vol_spike
            
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
            # Long - exit conditions
            # Stoploss at L4 (1.1 * range below previous close)
            stop_loss = l4
            # Take profit at opposite extreme or trend reversal
            take_profit = h3 if is_uptrend else l3
            
            if close_val <= stop_loss or close_val >= take_profit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit conditions
            # Stoploss at H4 (1.1 * range above previous close)
            stop_loss = h4
            # Take profit at opposite extreme or trend reversal
            take_profit = l3 if is_uptrend else h3
            
            if close_val >= stop_loss or close_val <= take_profit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_Pivot_VolumeRegime_1wTrend"
timeframe = "1d"
leverage = 1.0