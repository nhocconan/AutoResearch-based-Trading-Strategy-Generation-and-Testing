#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme + 1d EMA200 trend filter + volume spike
    # Long when: Williams %R(14) < -80 (oversold) AND close > 1d EMA200 (uptrend) AND volume > 1.5x 20-bar avg
    # Short when: Williams %R(14) > -20 (overbought) AND close < 1d EMA200 (downtrend) AND volume > 1.5x 20-bar avg
    # Exit when: Williams %R crosses above -50 (for long) or below -50 (for short)
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years.
    # Williams %R identifies exhaustion; 1d EMA200 filters counter-trend trades; volume confirms conviction.
    # Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    williams_window = 14
    highest_high = pd.Series(high).rolling(window=williams_window, min_periods=williams_window).max().values
    lowest_low = pd.Series(low).rolling(window=williams_window, min_periods=williams_window).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(williams_window, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        williams_exit_long = williams_r[i] > -50
        williams_exit_short = williams_r[i] < -50
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = williams_oversold and (close[i] > ema_200_1d_aligned[i]) and volume_confirmed[i] and position != 1
        short_entry = williams_overbought and (close[i] < ema_200_1d_aligned[i]) and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and williams_exit_long)
        exit_short = (position == -1 and williams_exit_short)
        
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

name = "6h_1d_williamsr_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0