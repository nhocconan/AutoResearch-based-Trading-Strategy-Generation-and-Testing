#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R mean reversion + 1d EMA(200) trend filter + volume spike
    # Long when: Williams %R(14) < -80 (oversold) AND price > 1d EMA200 AND volume > 1.5x 20-bar avg volume
    # Short when: Williams %R(14) > -20 (overbought) AND price < 1d EMA200 AND volume > 1.5x 20-bar avg volume
    # Exit when: Williams %R crosses above -50 (long exit) or below -50 (short exit)
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years.
    # Williams %R identifies overextended moves; 1d EMA200 filters counter-trend trades;
    # Volume spike confirms reversal validity. Works in bull (buy dips) and bear (sell rallies).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA(200) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(200) trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 12h timeframe (wait for completed 1d bar)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Williams %R(14) on 12h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        exit_long = williams_r[i] > -50  # exit long when %R crosses above -50
        exit_short = williams_r[i] < -50  # exit short when %R crosses below -50
        
        # 1d EMA200 trend filter
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = oversold and uptrend and volume_confirmed[i] and position != 1
        short_entry = overbought and downtrend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long_signal = (position == 1 and exit_long)
        exit_short_signal = (position == -1 and exit_short)
        
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

name = "12h_1d_williamsr_ema200_volume_v1"
timeframe = "12h"
leverage = 1.0