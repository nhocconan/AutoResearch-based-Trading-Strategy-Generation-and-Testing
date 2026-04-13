#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R mean reversion with 1w trend filter and volume spike confirmation.
    # Williams %R identifies overbought/oversold conditions; weekly trend ensures mean reversion trades align with higher timeframe direction.
    # Volume spike confirms exhaustion and reduces false signals. Works in ranging and trending markets via trend filter.
    # Target: 50-150 total trades over 4 years (12-37/year). Discrete size 0.25 to minimize fees.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Williams %R and volume (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 6h Williams %R (14-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_6h) / (highest_high - lowest_low)
    
    # Calculate 6h volume mean (20-period) with min_periods
    volume_6h_series = pd.Series(df_6h['volume'].values)
    vol_ma_20_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h volume for spike detection
        volume_6h_raw = df_6h['volume'].values
        vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h_raw)
        
        # Volume filter: current 6h volume > 1.8 * 20-period mean (volume spike for exhaustion)
        volume_confirmation = vol_6h_aligned[i] > 1.8 * vol_ma_aligned[i]
        
        # Trend filter: price above/below weekly EMA50
        price_above_weekly_ema = close[i] > ema_50_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_aligned[i]
        
        # Williams %R mean reversion conditions
        williams_oversold = williams_r_aligned[i] < -80  # Oversold
        williams_overbought = williams_r_aligned[i] > -20  # Overbought
        
        # Entry conditions: mean reversion with volume spike and trend alignment
        long_entry = williams_oversold and volume_confirmation and price_above_weekly_ema
        short_entry = williams_overbought and volume_confirmation and price_below_weekly_ema
        
        # Exit conditions: Williams %R returns to neutral zone or loss of volume confirmation
        williams_neutral = williams_r_aligned[i] > -50 and williams_r_aligned[i] < -50  # This is impossible, fix below
        # Corrected: exit when Williams %R returns to midpoint (-50) or volume confirmation lost
        long_exit = williams_r_aligned[i] > -50 or not volume_confirmation
        short_exit = williams_r_aligned[i] < -50 or not volume_confirmation
        
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

name = "6h_6h_1w_williamsr_mean_reversion_volume_v1"
timeframe = "6h"
leverage = 1.0