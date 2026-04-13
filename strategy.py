#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams Alligator + 1w trend filter + volume confirmation
    # Long when: price > Alligator Jaw (13-period SMMA shifted 8) AND price > 1w EMA(34) (uptrend) AND volume > 1.5x 20-bar avg volume
    # Short when: price < Alligator Lips (8-period SMMA shifted 5) AND price < 1w EMA(34) (downtrend) AND volume > 1.5x 20-bar avg volume
    # Exit when: price crosses Alligator Teeth (8-period SMMA) OR adverse 1w EMA(34) crossover
    # Uses discrete sizing (0.25) targeting 30-100 trades over 4 years.
    # Williams Alligator identifies trendless markets; only trade when aligned with 1w trend.
    # Volume confirmation reduces false signals in choppy 1d markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams Alligator components (SMMA with shifts)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price_1d = (high_1d + low_1d) / 2
    jaw_raw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift right by 8 (future data -> past alignment)
    jaw[:8] = np.nan  # first 8 values invalid due to shift
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift right by 5
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # shift right by 3
    lips[:3] = np.nan
    
    # Align Alligator components to 1d timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Get 1w data for EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA(34)
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Williams Alligator conditions
        price_above_jaw = close[i] > jaw_aligned[i]
        price_below_lips = close[i] < lips_aligned[i]
        price_above_teeth = close[i] > teeth_aligned[i]
        price_below_teeth = close[i] < teeth_aligned[i]
        
        # 1w EMA(34) trend filter
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Entry conditions with volume confirmation
        long_entry = price_above_jaw and uptrend and volume_confirmed[i] and position != 1
        short_entry = price_below_lips and downtrend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (price_below_teeth or not uptrend))
        exit_short = (position == -1 and (price_above_teeth or not downtrend))
        
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

name = "1d_1w_williams_alligator_ema_volume_v1"
timeframe = "1d"
leverage = 1.0