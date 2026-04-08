#!/usr/bin/env python3
# 6h_camarilla_pivot_1d_trend_volume_v1
# Hypothesis: 6h Camarilla pivot breakout/fade with 1d trend filter and volume confirmation.
# Uses 6h Camarilla levels (H4/L4 breakout, H3/L3 fade) filtered by 1d EMA trend.
# Volume filter ensures momentum behind moves. Works in bull/bear by following higher timeframe trend.
# Target: 15-30 trades/year via strict Camarilla conditions + trend alignment + volume filter.

name = "6h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h range for Camarilla calculation (using prior bar's range)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels: based on previous bar's range
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # H2 = close + 1.0 * (high - low)
    # L2 = close - 1.0 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    range_prev = prev_high - prev_low
    h4 = prev_close + 1.5 * range_prev
    l4 = prev_close - 1.5 * range_prev
    h3 = prev_close + 1.25 * range_prev
    l3 = prev_close - 1.25 * range_prev
    h2 = prev_close + 1.0 * range_prev
    l2 = prev_close - 1.0 * range_prev
    h1 = prev_close + 0.5 * range_prev
    l1 = prev_close - 0.5 * range_prev
    
    # 1d EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 20-period average volume for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(h4[i]) or np.isnan(l4[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or
            np.isnan(h2[i]) or np.isnan(l2[i]) or np.isnan(h1[i]) or np.isnan(l1[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        volume_filter = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price closes below H3 (fade level) OR below L4 (stop)
            if close[i] < h3[i] or close[i] < l4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price closes above L3 (fade level) OR above H4 (stop)
            if close[i] > l3[i] or close[i] > h4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Determine trend from 1d EMA
            uptrend = close[i] > ema_1d_aligned[i]
            downtrend = close[i] < ema_1d_aligned[i]
            
            # In uptrend: look for breakouts above H4 or pullbacks to L3
            # In downtrend: look for breakdowns below L4 or pullbacks to H3
            
            # Long breakout: price breaks above H4 with volume in uptrend
            if (close[i] > h4[i] and 
                volume_filter and 
                uptrend):
                position = 1
                signals[i] = 0.25
            # Long fade: price pulls back to L3 with volume in uptrend
            elif (close[i] <= l3[i] and 
                  volume_filter and 
                  uptrend and
                  close[i] > l4[i]):  # Not broken below L4
                position = 1
                signals[i] = 0.25
            # Short breakdown: price breaks below L4 with volume in downtrend
            elif (close[i] < l4[i] and 
                  volume_filter and 
                  downtrend):
                position = -1
                signals[i] = -0.25
            # Short fade: price pulls back to H3 with volume in downtrend
            elif (close[i] >= h3[i] and 
                  volume_filter and 
                  downtrend and
                  close[i] < h4[i]):  # Not broken above H4
                position = -1
                signals[i] = -0.25
    
    return signals