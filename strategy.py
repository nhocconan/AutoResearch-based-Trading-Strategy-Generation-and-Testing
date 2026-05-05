#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when: Price breaks above Donchian upper (20) AND close > 1w EMA50 AND volume > 1.5x 20-period avg volume
# Short when: Price breaks below Donchian lower (20) AND close < 1w EMA50 AND volume > 1.5x 20-period avg volume
# Exit when: Price touches Donchian middle (mean of upper/lower) OR opposite breakout occurs
# Donchian breakout captures sustained momentum; 1w EMA50 ensures alignment with weekly trend
# Volume confirmation filters false breakouts; works in both bull/bear by trading with weekly trend
# Target: 50-100 total trades over 4 years (12-25/year) with discrete sizing 0.25

name = "1d_Donchian20_1wEMA50_Volume"
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
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA(50)
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20) on 1d
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_donch = highest_20
    lower_donch = lowest_20
    middle_donch = (upper_donch + lower_donch) / 2.0
    
    # Calculate 20-period average volume for confirmation
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_donch[i]) or 
            np.isnan(lower_donch[i]) or np.isnan(avg_vol_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * avg_vol_20[i]
        
        if position == 0:
            # Long: Break above upper Donchian with weekly trend up and volume confirmation
            if close[i] > upper_donch[i] and close[i] > ema_50_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with weekly trend down and volume confirmation
            elif close[i] < lower_donch[i] and close[i] < ema_50_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches middle Donchian OR breaks below lower (reversal)
            if close[i] <= middle_donch[i] or close[i] < lower_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches middle Donchian OR breaks above upper (reversal)
            if close[i] >= middle_donch[i] or close[i] > upper_donch[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals