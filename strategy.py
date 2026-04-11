#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1w Camarilla pivot with volume confirmation and 1d trend filter
# Long when price touches Camarilla L3 support + volume > 1.5x average + 1d trend up
# Short when price touches Camarilla H3 resistance + volume > 1.5x average + 1d trend down
# Exit when price reaches Camarilla H4/L4 or trend reverses
# Designed for 12-37 trades/year on 12h timeframe with mean reversion in range and trend following in breakouts

name = "12h_1w_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivot levels (based on previous week)
    # Camarilla formulas: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    # We use the previous week's data to calculate levels for current week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot levels for each week (using previous week's data)
    camarilla_h4 = close_1w + 1.5 * (high_1w - low_1w)
    camarilla_h3 = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_l3 = close_1w - 1.1 * (high_1w - low_1w)
    camarilla_l4 = close_1w - 1.5 * (high_1w - low_1w)
    
    # Align to 12h timeframe (wait for weekly bar to close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume filter (using 12h data)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price relative to 1d EMA50
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: price touches Camarilla H3/L3 with volume and trend confirmation
        # Use small epsilon for touch detection due to floating point precision
        epsilon = 0.0001 * close[i]
        touches_h3 = abs(high[i] - camarilla_h3_aligned[i]) <= epsilon
        touches_l3 = abs(low[i] - camarilla_l3_aligned[i]) <= epsilon
        
        long_entry = touches_l3 and volume_filter and is_uptrend
        short_entry = touches_h3 and volume_filter and is_downtrend
        
        # Exit conditions: price reaches H4/L4 or trend reverses
        long_exit = (low[i] <= camarilla_l4_aligned[i] + epsilon) or (not is_uptrend)
        short_exit = (high[i] >= camarilla_h4_aligned[i] - epsilon) or (not is_downtrend)
        
        # Priority: entry > exit > hold
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
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals