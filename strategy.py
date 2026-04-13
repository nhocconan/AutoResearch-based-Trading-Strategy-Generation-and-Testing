#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1d EMA filter and volume confirmation.
Williams %R identifies overbought/oversold conditions (above -20 = overbought, below -80 = oversold).
In trending markets (price above/below 1d EMA), we fade extremes: short when Williams %R > -20 and price > 1d EMA,
long when Williams %R < -80 and price < 1d EMA.
Volume confirmation requires 6h volume > 1.5x 20-period average to avoid low-volume false signals.
Designed for mean reversion in trends, works in both bull (fade rallies) and bear (fade bounces).
Target: 60-180 trades over 4 years (15-45/year) to balance opportunity and fee cost.
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
    
    # Get 1d data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R (14-period) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 6h volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Williams %R extremes + EMA trend filter + volume confirmation
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        vol_confirm = vol_spike[i]
        
        long_entry = williams_oversold and price_below_ema and vol_confirm
        short_entry = williams_overbought and price_above_ema and vol_confirm
        
        # Exit when Williams %R returns to neutral range (-50) or opposite extreme
        exit_long = position == 1 and williams_r[i] > -50
        exit_short = position == -1 and williams_r[i] < -50
        
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