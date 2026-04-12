#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot long/short with 1w EMA200 trend filter + volume spike confirmation
    # Only take long at Camarilla L3 in bullish 1w trend (price > EMA200) and short at H3 in bearish trend
    # Volume confirmation: volume > 2.0 * 20-period average to filter false signals
    # Discrete sizing 0.25 to minimize fee churn. Target: 15-25 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's high, low, close)
    # We'll calculate these for each bar using the previous completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H3, H4, L3, L4
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align 1d Camarilla levels to 1h timeframe (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1w trend
        bullish_trend = close[i] > ema200_1w_aligned[i]
        bearish_trend = close[i] < ema200_1w_aligned[i]
        
        # Entry logic: Camarilla touch with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long entry: price touches or breaks below L3 in bullish 1w trend (mean reversion long)
        if bullish_trend:
            long_entry = (low[i] <= camarilla_l3_aligned[i]) and volume_spike[i]
        # Short entry: price touches or breaks above H3 in bearish 1w trend (mean reversion short)
        elif bearish_trend:
            short_entry = (high[i] >= camarilla_h3_aligned[i]) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or trend reversal
        long_exit = (bearish_trend and high[i] >= camarilla_h3_aligned[i]) or \
                    (close[i] > camarilla_h3_aligned[i]) or \
                    (not bullish_trend and not bearish_trend)
        short_exit = (bullish_trend and low[i] <= camarilla_l3_aligned[i]) or \
                     (close[i] < camarilla_l3_aligned[i]) or \
                     (not bullish_trend and not bearish_trend)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_camarilla_pivot_meanrev_trend_filter_v1"
timeframe = "1d"
leverage = 1.0