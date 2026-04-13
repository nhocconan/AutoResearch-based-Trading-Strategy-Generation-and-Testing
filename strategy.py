#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h volume filter and 1d trend filter
    # Long when: price breaks above Camarilla H3 AND 4h volume > 1.5x 20-bar avg AND price > 1d EMA(50) (uptrend)
    # Short when: price breaks below Camarilla L3 AND 4h volume > 1.5x 20-bar avg AND price < 1d EMA(50) (downtrend)
    # Exit when: price crosses Camarilla pivot point (PP) OR adverse 1d EMA(50) crossover
    # Uses discrete sizing (0.20) targeting 60-150 trades over 4 years.
    # Works in bull/bear via 1d EMA(50) trend filter preventing counter-trend trades.
    # Volume confirmation reduces false breakouts in choppy markets.
    # Session filter (08-20 UTC) reduces noise trades.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Get 4h data for volume filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 4h volume confirmation: volume > 1.5x 20-bar average volume
    volume_4h = df_4h['volume'].values
    avg_volume_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_confirmed_4h = volume_4h > (1.5 * avg_volume_4h)
    volume_confirmed_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_confirmed_4h)
    
    # Calculate Camarilla pivot points for 1h timeframe
    # Based on previous period's high, low, close
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Camarilla calculations
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_range = prev_high - prev_low
    camarilla_h3 = camarilla_pp + (camarilla_range * 1.1 / 4)
    camarilla_l3 = camarilla_pp - (camarilla_range * 1.1 / 4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            # Outside session: flatten position
            if position == 1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(camarilla_pp[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_confirmed_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions
        breakout_up = close[i] > camarilla_h3[i]  # Break above H3
        breakout_down = close[i] < camarilla_l3[i]  # Break below L3
        
        # 1d EMA(50) trend filter
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        # Entry conditions with volume confirmation and session filter already applied
        long_entry = breakout_up and uptrend and volume_confirmed_4h_aligned[i] and position != 1
        short_entry = breakout_down and downtrend and volume_confirmed_4h_aligned[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < camarilla_pp[i] or not uptrend))
        exit_short = (position == -1 and (close[i] > camarilla_pp[i] or not downtrend))
        
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

name = "1h_4h_1d_camarilla_volume_trend_v1"
timeframe = "1h"
leverage = 1.0