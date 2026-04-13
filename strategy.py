#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d trend filter
    # Long when: price breaks above H3 (bullish breakout) AND price > 1d EMA50 (uptrend)
    # Short when: price breaks below L3 (bearish breakdown) AND price < 1d EMA50 (downtrend)
    # Exit when: price returns to pivot level (mean reversion) OR adverse 1d EMA50 crossover
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via 1d EMA50 trend filter preventing counter-trend trades.
    # Camarilla pivots provide structure in ranging markets, trend filter avoids whipsaws.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for pivot calculation and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous 1d bar (H3, L3, pivot)
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    # pivot = (high+low+close)/3
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    
    # Align HTF levels to LTF (wait for completed 1d bar)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla conditions
        breakout_long = close[i] > h3_aligned[i]  # Price breaks above H3
        breakdown_short = close[i] < l3_aligned[i]  # Price breaks below L3
        return_to_pivot = abs(close[i] - pivot_aligned[i]) < 0.001 * close[i]  # Near pivot (0.1%)
        
        # 1d EMA50 trend filter
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        long_entry = breakout_long and uptrend and position != 1
        short_entry = breakdown_short and downtrend and position != -1
        
        # Exit conditions
        exit_long = return_to_pivot or (position == 1 and not uptrend)
        exit_short = return_to_pivot or (position == -1 and not downtrend)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_trend_v1"
timeframe = "12h"
leverage = 1.0