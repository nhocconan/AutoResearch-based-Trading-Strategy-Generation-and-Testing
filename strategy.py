# 144300
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ThreeBarReversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume MA for confirmation
    volume_1d = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = volume_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Three-bar reversal pattern detection
    # Bullish reversal: 3 consecutive lower lows followed by a higher close
    # Bearish reversal: 3 consecutive higher highs followed by a lower close
    low_series = pd.Series(low)
    high_series = pd.Series(high)
    
    # Three consecutive lower lows (for bullish reversal)
    lower_low_1 = low_series.shift(1) > low_series.shift(2)
    lower_low_2 = low_series.shift(2) > low_series.shift(3)
    three_lower_lows = lower_low_1 & lower_low_2
    
    # Three consecutive higher highs (for bearish reversal)
    higher_high_1 = high_series.shift(1) < high_series.shift(2)
    higher_high_2 = high_series.shift(2) < high_series.shift(3)
    three_higher_highs = higher_high_1 & higher_high_2
    
    # Shift to avoid look-ahead: signal based on completed bar
    three_lower_lows_shifted = three_lower_lows.shift(1).fillna(False).values
    three_higher_highs_shifted = three_higher_highs.shift(1).fillna(False).values
    
    # Volume confirmation: current volume > 1.5x 1d average volume
    vol_ok = volume > 1.5 * vol_ma20_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bullish reversal: 3 lower lows + higher close + volume + above 1d EMA
            if (three_lower_lows_shifted[i] and 
                close[i] > close[i-1] and 
                vol_ok[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Bearish reversal: 3 higher highs + lower close + volume + below 1d EMA
            elif (three_higher_highs_shifted[i] and 
                  close[i] < close[i-1] and 
                  vol_ok[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below entry or reversal signal
            if close[i] < close[i-1] and three_higher_highs_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above entry or reversal signal
            if close[i] > close[i-1] and three_lower_lows_shifted[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals