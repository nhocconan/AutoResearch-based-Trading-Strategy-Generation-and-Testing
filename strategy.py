#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation
    # Camarilla levels (R3/S3 for mean reversion, R4/S4 for breakout) derived from 1d OHLC
    # Trade breakouts at R4/S4 with volume spike confirmation (>1.5x 20-period avg volume)
    # Use R3/S3 for mean reversion exits to avoid giving back profits in chop
    # Works in bull/bear by adapting to volatility via volume confirmation
    # Target: 12-37 trades/year per symbol (~50-150 total over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and volume context (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align 1d Camarilla levels to 6h timeframe (wait for completed 1d bar)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions at R4/S4
        breakout_long = close[i] > r4_aligned[i]
        breakout_short = close[i] < s4_aligned[i]
        
        # Mean reversion conditions at R3/S3 (for exits)
        mean_revert_long = close[i] < r3_aligned[i]  # Exit long when price drops below R3
        mean_revert_short = close[i] > s3_aligned[i]  # Exit short when price rises above S3
        
        # Entry conditions: breakout with volume confirmation
        long_entry = breakout_long and volume_filter[i]
        short_entry = breakout_short and volume_filter[i]
        
        # Exit conditions: opposite breakout OR mean reversion at R3/S3
        long_exit = (close[i] < s4_aligned[i]) or mean_revert_long
        short_exit = (close[i] > r4_aligned[i]) or mean_revert_short
        
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

name = "6h_1d_camarilla_breakout_vol_filter_v1"
timeframe = "6h"
leverage = 1.0