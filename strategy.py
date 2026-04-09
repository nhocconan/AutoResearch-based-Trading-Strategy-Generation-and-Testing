#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1w volume confirmation and trend filter
# In trending markets (price > weekly EMA200): breakout above/below Camarilla H3/L3 levels with volume spike
# In ranging markets (price near weekly EMA200): mean reversion at H3/L3 levels
# Uses discrete position sizing 0.25 to target ~12-37 trades/year and minimize fee drag
# Works in bull/bear markets: breakout captures trends, mean reversion captures ranging markets

name = "12h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 1w average volume (20-period)
    volume_1w_series = pd.Series(volume_1w)
    avg_volume_1w = volume_1w_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d data for Camarilla pivots (more frequent than 1w)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on prior day to avoid look-ahead)
    range_1d = high_1d - low_1d
    h3_1d = close_1d + 1.1 * range_1d
    l3_1d = close_1d - 1.1 * range_1d
    h4_1d = close_1d + 1.5 * range_1d
    l4_1d = close_1d - 1.5 * range_1d
    
    # Align 1w indicators to 12h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    avg_volume_1w_aligned = align_htf_to_ltf(prices, df_1w, avg_volume_1w)
    
    # Align 1d indicators to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Pre-compute volume confirmation array
    volume_confirmed = volume > 2.0 * avg_volume_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(avg_volume_1w_aligned[i]) or
            np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly EMA200
        uptrend = close[i] > ema200_1w_aligned[i]
        downtrend = close[i] < ema200_1w_aligned[i]
        
        if position == 1:  # Long position
            if uptrend:
                # Exit long if price breaks below H3 or trend turns down
                if close[i] < h3_1d_aligned[i] or downtrend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # ranging or downtrend
                # Exit long if price rises above H4 or drops below L3
                if close[i] > h4_1d_aligned[i] or close[i] < l3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                
        elif position == -1:  # Short position
            if downtrend:
                # Exit short if price breaks above L3 or trend turns up
                if close[i] > l3_1d_aligned[i] or uptrend:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # ranging or uptrend
                # Exit short if price drops below L4 or rises above H3
                if close[i] < l4_1d_aligned[i] or close[i] > h3_1d_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            if uptrend:
                # Enter long on breakout above H3 with volume confirmation
                if close[i] > h3_1d_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short on breakdown below L3 with volume confirmation
                elif close[i] < l3_1d_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
            elif downtrend:
                # Enter short on breakdown below L3 with volume confirmation
                if close[i] < l3_1d_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
                # Enter long on breakout above H3 with volume confirmation
                elif close[i] > h3_1d_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
            else:  # ranging market
                # Mean reversion: buy near L3, sell near H3
                if close[i] <= l3_1d_aligned[i] and volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= h3_1d_aligned[i] and volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals