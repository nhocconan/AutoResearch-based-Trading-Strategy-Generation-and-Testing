#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1-week High-Low breakout + volume surge + volatility filter.
# Uses tight entry conditions (volatility > 1.5x median of last 30 days, volume > 2.0x average volume) to limit trades to ~10-25/year.
# Works in bull/bear markets by capturing breakouts with institutional volume confirmation.
# Designed for low trade frequency to minimize fee drag while maintaining edge in ranging/trending regimes.

name = "1d_1w_breakout_volume_volatility_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w High and Low (simple highest high and lowest low over the week)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Align 1w High and Low to 1d timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Calculate 1d volatility (ATR-like: average true range over 10 days)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first period
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure volatility and volume averages are valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(high_1w_aligned[i]) or np.isnan(low_1w_aligned[i]) or 
            np.isnan(atr_10[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility filter: current volatility > 1.5 * median of last 30 periods
        vol_filter = atr_10[i] > 1.5 * np.nanmedian(atr_10[max(0, i-30):i])
        
        # Volume filter: current volume > 2.0 * average volume (higher threshold for fewer trades)
        vol_surge = volume[i] > 2.0 * vol_avg_20[i]
        
        # Entry conditions: price breaks through 1w High/Low with volatility and volume surge
        long_entry = (high[i] > high_1w_aligned[i] and vol_filter and vol_surge)
        short_entry = (low[i] < low_1w_aligned[i] and vol_filter and vol_surge)
        
        # Exit conditions: price returns to the opposite extreme (mean reversion)
        # Long exit when price returns to weekly low
        # Short exit when price returns to weekly high
        long_exit = low[i] < low_1w_aligned[i] if not np.isnan(low_1w_aligned[i]) else False
        short_exit = high[i] > high_1w_aligned[i] if not np.isnan(high_1w_aligned[i]) else False
        
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