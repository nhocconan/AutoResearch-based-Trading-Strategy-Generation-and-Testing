#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams %R extreme + 1w trend filter + volume spike
    # Long: Williams %R < -80 (oversold) AND price > 1w EMA20 (uptrend) AND 1d volume > 1.5 * 20-day average
    # Short: Williams %R > -20 (overbought) AND price < 1w EMA20 (downtrend) AND 1d volume > 1.5 * 20-day average
    # Exit: Williams %R crosses above -50 (for long) or below -50 (for short) OR volume drops below average
    # Using 1d for entry/exit, 1w for trend filter to avoid look-ahead
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 15-25 trades/year (~60-100 over 4 years) to stay within fee drag limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for EMA20 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d volume spike filter: volume > 1.5 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d indicators to 1d timeframe (no additional delay needed for same timeframe)
    williams_r_aligned = williams_r  # already 1d
    volume_spike_aligned = volume_spike.astype(float)  # already 1d
    
    # Align 1w EMA20 to 1d timeframe (wait for completed 1w bar)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extremes
        williams_r_val = williams_r_aligned[i]
        oversold = williams_r_val < -80
        overbought = williams_r_val > -20
        exit_long = williams_r_val > -50  # crosses above -50
        exit_short = williams_r_val < -50  # crosses below -50
        
        # Volume confirmation: 1d volume spike
        vol_confirmed = volume_spike_aligned[i] > 0.5  # boolean as float
        
        # Trend filter: 1w EMA20
        price_vs_ema = close[i] > ema_20_1w_aligned[i]  # True if price above 1w EMA20
        
        # Entry logic: Williams %R extreme + volume spike + trend alignment
        long_entry = oversold and vol_confirmed and price_vs_ema
        short_entry = overbought and vol_confirmed and (not price_vs_ema)
        
        # Exit logic: Williams %R crosses -50 OR volume drops below average
        long_exit = exit_long or (not vol_confirmed)
        short_exit = exit_short or (not vol_confirmed)
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_williamsr_volume_trend_v1"
timeframe = "1d"
leverage = 1.0