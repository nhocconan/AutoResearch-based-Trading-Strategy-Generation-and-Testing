# 12h_1d_1w_camarilla_pivot_volume_v1
# Hypothesis: Using daily Camarilla pivot levels (support/resistance) with weekly trend filter (50-day EMA) and volume confirmation creates high-probability mean-reversion entries. The Camarilla levels provide mathematically derived support/resistance zones that work in both trending and ranging markets. Weekly EMA filter ensures we only trade in the direction of the higher timeframe trend, while volume confirmation filters out low-activity false breakouts. Designed for 12h timeframe to target 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    # Formula: 
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_h2 = close_1d + 0.5 * (high_1d - low_1d)
    camarilla_h1 = close_1d + 0.25 * (high_1d - low_1d)
    camarilla_l1 = close_1d - 0.25 * (high_1d - low_1d)
    camarilla_l2 = close_1d - 0.5 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.0 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Get weekly data for trend filter (50-day EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-week EMA for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: volume > 1.5x average of last 10 periods
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(h4_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_trend_up = close[i] > ema50_1w_aligned[i]
        weekly_trend_down = close[i] < ema50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price reaches H3 or weekly trend turns down
            if close[i] >= h3_aligned[i] or not weekly_trend_up:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price reaches L3 or weekly trend turns up
            if close[i] <= l3_aligned[i] or not weekly_trend_down:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price touches L3/L4 with weekly uptrend and volume confirmation
            if (weekly_trend_up and vol_confirm[i] and 
                (close[i] <= l3_aligned[i] or close[i] <= l4_aligned[i])):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches H3/H4 with weekly downtrend and volume confirmation
            elif (weekly_trend_down and vol_confirm[i] and 
                  (close[i] >= h3_aligned[i] or close[i] >= h4_aligned[i])):
                position = -1
                signals[i] = -0.25
    
    return signals