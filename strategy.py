# 1d_Weekly_High_Low_Breakout_With_Volume_Confirmation
# Hypothesis: Breakouts above weekly high or below weekly low on daily chart, confirmed by volume spikes.
# Weekly levels act as strong support/resistance; breakouts capture momentum moves.
# Volume confirmation filters false breakouts. Designed for low trade frequency (7-25/year) to avoid fee drag.
# Works in both bull (catch breakouts) and bear (catch breakdowns) markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for high/low levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Align weekly high/low to daily (wait for weekly bar to close)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, high_1w)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Volume spike: >2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 5)  # Warmup for volume MA and weekly alignment
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_high = weekly_high_aligned[i]
        weekly_low = weekly_low_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above weekly high with volume spike
            if price > weekly_high and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low with volume spike
            elif price < weekly_low and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price falls back below weekly high (failed breakout) or opposite signal
            if price < weekly_high:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price rises back above weekly low (failed breakdown) or opposite signal
            if price > weekly_low:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Weekly_High_Low_Breakout_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0