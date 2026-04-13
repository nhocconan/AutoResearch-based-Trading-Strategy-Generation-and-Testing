#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation
    # Long when: price breaks above H3 (bullish breakout) AND 1w EMA20 uptrend AND volume > 1.5x 20-period average
    # Short when: price breaks below L3 (bearish breakout) AND 1w EMA20 downtrend AND volume > 1.5x 20-period average
    # Exit when: price returns to Pivot Point (mean reversion) OR adverse 1w EMA20 crossover
    # Uses discrete sizing (0.25) targeting 30-100 trades over 4 years.
    # Works in bull/bear via 1w EMA20 trend filter preventing counter-trend trades.
    # Camarilla levels provide institutional support/resistance with high probability reactions.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels calculation
    range_1d = prev_high - prev_low
    camarilla_h3 = prev_close + range_1d * 1.1 / 4
    camarilla_l3 = prev_close - range_1d * 1.1 / 4
    camarilla_h4 = prev_close + range_1d * 1.1 / 2
    camarilla_l4 = prev_close - range_1d * 1.1 / 2
    camarilla_h5 = prev_close + range_1d * 1.1
    camarilla_l5 = prev_close - range_1d * 1.1
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 1d timeframe
    h3_1d = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_1d = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_1d = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_1d = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h5_1d = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    l5_1d = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    pivot_1d = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA20
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1d[i]) or np.isnan(l3_1d[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter
        uptrend = close[i] > ema20_1w_aligned[i]
        downtrend = close[i] < ema20_1w_aligned[i]
        
        # Breakout conditions (using today's price vs yesterday's levels)
        long_breakout = close[i] > h3_1d[i] and volume_confirm[i]
        short_breakout = close[i] < l3_1d[i] and volume_confirm[i]
        
        # Mean reversion exit conditions
        exit_long = close[i] < pivot_1d[i]  # Return to pivot
        exit_short = close[i] > pivot_1d[i]  # Return to pivot
        
        # Entry conditions
        long_entry = long_breakout and uptrend and position != 1
        short_entry = short_breakout and downtrend and position != -1
        
        # Exit conditions
        exit_long_signal = exit_long or (position == 1 and not uptrend)
        exit_short_signal = exit_short or (position == -1 and not downtrend)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long_signal:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short_signal:
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

name = "1d_1w_camarilla_breakout_volume_trend_v2"
timeframe = "1d"
leverage = 1.0