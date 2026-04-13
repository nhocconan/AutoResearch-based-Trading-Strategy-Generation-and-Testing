#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w EMA(34) trend filter and volume confirmation
    # Long when: price breaks above Camarilla H3 level AND price > 1w EMA(34) (uptrend) AND volume > 2x 20-bar avg volume
    # Short when: price breaks below Camarilla L3 level AND price < 1w EMA(34) (downtrend) AND volume > 2x 20-bar avg volume
    # Exit when: price crosses Camarilla pivot point (PP) OR adverse 1w EMA(34) crossover
    # Uses discrete sizing (0.25) targeting 30-100 trades over 4 years.
    # Works in bull/bear via 1w EMA(34) trend filter preventing counter-trend trades.
    # Volume confirmation reduces false breakouts in choppy markets.
    # Camarilla pivots derived from prior day's range work well in both trending and ranging markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Camarilla pivot levels from prior 1d bar
    # PP = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    # Shift 1d data by 1 to get prior day's OHLC (avoid look-ahead)
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    
    # Align prior 1d OHLC to lower timeframe
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    prior_close_aligned = align_htf_to_ltf(prices, df_1d, prior_close)
    
    # Calculate Camarilla levels
    PP = (prior_high_aligned + prior_low_aligned + prior_close_aligned) / 3
    range_val = prior_high_aligned - prior_low_aligned
    
    R3 = prior_close_aligned + (range_val * 1.1 / 4)
    R4 = prior_close_aligned + (range_val * 1.1 / 2)
    S3 = prior_close_aligned - (range_val * 1.1 / 4)
    S4 = prior_close_aligned - (range_val * 1.1 / 2)
    
    # Calculate volume confirmation: volume > 2x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(PP[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > R3[i-1]  # Break above R3 (strong resistance)
        breakout_down = close[i] < S3[i-1]  # Break below S3 (strong support)
        
        # 1w EMA(34) trend filter
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
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

name = "1d_1w_camarilla_pivot_breakout_volume_v1"
timeframe = "1d"
leverage = 1.0