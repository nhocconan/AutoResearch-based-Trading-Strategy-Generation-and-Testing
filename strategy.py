#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA trend filter and volume confirmation.
Williams %R identifies overbought/oversold conditions (above -20 = overbought, below -80 = oversold).
In trending markets (price above/below 1d EMA50), we fade extremes only when aligned with trend:
- Long: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.5x 20-period average
- Short: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.5x 20-period average
Exit when Williams %R returns to neutral territory (-50) or reverses.
This captures mean reversion within trends, avoiding counter-trend trades. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 1-day volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        neutral_return = abs(williams_r[i]) < 50  # Return to near -50 level
        
        # Trend filter: price vs 1d EMA50
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike_aligned[i] > 0.5
        
        # Entry conditions
        long_entry = oversold and uptrend and vol_confirm
        short_entry = overbought and downtrend and vol_confirm
        
        # Exit conditions: return to neutral or reverse signal
        exit_long = (position == 1) and (neutral_return or overbought)
        exit_short = (position == -1) and (neutral_return or oversold)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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

name = "6h_1d_williams_r_ema_volume"
timeframe = "6h"
leverage = 1.0