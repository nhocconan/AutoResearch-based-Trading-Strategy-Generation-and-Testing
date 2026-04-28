#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1w trend filter and volume confirmation.
# Williams %R < -80 = oversold (long), > -20 = overbought (short) on 6h.
# Trend filter: price > 1w EMA34 for longs, price < 1w EMA34 for shorts.
# Volume confirmation: 6h volume > 1.5x 24-bar average (4d equivalent).
# Designed to capture mean reversals in extreme conditions while following the weekly trend.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

name = "6h_WilliamsR_1wEMA34_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 6h Williams %R (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # 6h volume spike: >1.5x 24-bar average volume
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 1.5 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Williams %R needs 14, volume MA needs 24, use 24 for safety
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(williams_r[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        williams_r_oversold = williams_r[i] < -80
        williams_r_overbought = williams_r[i] > -20
        
        # Trend filter: 1w EMA34 direction
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = williams_r_oversold and price_above_ema and vol_confirm
        short_entry = williams_r_overbought and price_below_ema and vol_confirm
        
        # Exit conditions: Williams %R returns to neutral zone (-50)
        long_exit = williams_r[i] > -50
        short_exit = williams_r[i] < -50
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals