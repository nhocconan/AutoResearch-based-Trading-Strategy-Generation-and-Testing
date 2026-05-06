#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with volume confirmation and ATR-based trailing stop
# Long when price breaks above 1w Donchian upper channel (20) AND volume > 1.5 * avg_volume(20) on 1d
# Short when price breaks below 1w Donchian lower channel (20) AND volume > 1.5 * avg_volume(20) on 1d
# Exit when price moves against position by 2.5 * ATR(14) from extreme favorable price
# Uses discrete sizing 0.25 to balance return and risk
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Donchian provides strong structural breakout levels with continuation probability
# Volume confirmation validates breakout strength while limiting false signals
# ATR trailing stop manages risk and allows trends to run
# Works in both bull (buy breakouts) and bear (sell breakdowns) markets

name = "1d_Donchian20_Volume_ATR_Trail"
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
    
    # Get 1w data ONCE before loop for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need at least 20 completed weekly bars for Donchian
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w Donchian channel (20-period)
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian levels to 1d timeframe (wait for completed 1w bar)
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1d
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate ATR(14) for trailing stop
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]  # First bar: no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_high = 0.0   # Highest price since entering long
    low_low = 0.0     # Lowest price since entering short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(atr[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_high = 0.0
                low_low = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Donchian upper, volume spike, in session
            if close[i] > upper_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                long_high = close[i]
            # Short: price breaks below 1w Donchian lower, volume spike, in session
            elif close[i] < lower_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                low_low = close[i]
        elif position == 1:
            # Update highest price since entry
            if close[i] > long_high:
                long_high = close[i]
            # Exit long: price moves against position by 2.5 * ATR(14) from extreme
            if close[i] <= long_high - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                long_high = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest price since entry
            if close[i] < low_low:
                low_low = close[i]
            # Exit short: price moves against position by 2.5 * ATR(14) from extreme
            if close[i] >= low_low + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                low_low = 0.0
            else:
                signals[i] = -0.25
    
    return signals