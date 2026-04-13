#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
    # Long when: price breaks above Camarilla H3 AND 1w close > 1w EMA200 (uptrend) AND volume > 2x 24-bar avg volume
    # Short when: price breaks below Camarilla L3 AND 1w close < 1w EMA200 (downtrend) AND volume > 2x 24-bar avg volume
    # Exit when: price crosses Camarilla pivot point (PP) OR adverse 1w EMA200 crossover
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Camarilla levels provide institutional support/resistance; 1w EMA200 filters counter-trend trades.
    # Volume confirmation reduces false breakouts. Works in bull/bear via 1w trend filter.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA200
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Use rolling window of 2 to get previous bar's OHLC
    prev_close = pd.Series(close).shift(1).values
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    
    # Camarilla calculations
    PP = (prev_high + prev_low + prev_close) / 3
    RANGE = prev_high - prev_low
    H3 = PP + (RANGE * 1.1 / 4)
    L3 = PP - (RANGE * 1.1 / 4)
    H4 = PP + (RANGE * 1.1 / 2)
    L4 = PP - (RANGE * 1.1 / 2)
    
    # Calculate volume confirmation: volume > 2x 24-bar average volume
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(PP[i]) or np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > H3[i]  # Break above H3
        breakout_down = close[i] < L3[i]  # Break below L3
        
        # 1w EMA200 trend filter
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and uptrend and volume_confirmed[i] and position != 1
        short_entry = breakout_down and downtrend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < PP[i] or not uptrend))
        exit_short = (position == -1 and (close[i] > PP[i] or not downtrend))
        
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

name = "12h_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0