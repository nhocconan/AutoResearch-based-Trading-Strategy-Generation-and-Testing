#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Ichimoku Cloud for trend direction and daily momentum for entry.
# The Ichimoku Cloud provides a robust trend filter that works in both bull and bear markets.
# Entries are taken when price crosses above/below the Cloud with volume confirmation.
# Exit when price re-enters the Cloud. Targets 15-25 trades/year (60-100 total over 4 years).
# Weekly timeframe for trend reduces whipsaw, daily for timely entries.
name = "1d_1w_IchimokuCloud_Momentum_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1d data for momentum and volume
    close_1d = close
    high_1d = high
    low_1d = low
    volume_1d = volume
    
    # Get weekly data for Ichimoku Cloud (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Ichimoku components on weekly data
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high_1w).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_1w).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high_1w).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_1w).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    period52_low = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    senkou_span_b = ((period52_high + period52_low) / 2)
    
    # Align Ichimoku components to daily timeframe
    # Note: Senkou spans are already shifted in calculation, so we align without additional shift
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b)
    
    # The Cloud is between Senkou Span A and Senkou Span B
    # For trend: price above Cloud = bullish, price below Cloud = bearish
    # We'll use the Cloud edges for entry/exit signals
    
    # Momentum: daily price change
    price_change = pd.Series(close_1d).diff().values
    
    # Volume filter: volume > 1.3 * 20-day average
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_1d > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen_aligned[i]) or np.isnan(kijun_sen_aligned[i]) or 
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i]) or
            np.isnan(price_change[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine Cloud boundaries
        cloud_top = max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        cloud_bottom = min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        if position == 0:
            # Long: price crosses above Cloud with positive momentum and volume
            if (close_1d[i] > cloud_top and 
                close_1d[i-1] <= cloud_top and  # crossed above
                price_change[i] > 0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below Cloud with negative momentum and volume
            elif (close_1d[i] < cloud_bottom and 
                  close_1d[i-1] >= cloud_bottom and  # crossed below
                  price_change[i] < 0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price re-enters Cloud (below cloud top)
            if close_1d[i] < cloud_top:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price re-enters Cloud (above cloud bottom)
            if close_1d[i] > cloud_bottom:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals