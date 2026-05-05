#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d volume confirmation and 4h trend filter
# Long when Williams %R(14) crosses above -80 (oversold bounce) AND volume > 1.8x 20-period average AND 4h EMA20 > EMA50 (uptrend)
# Short when Williams %R(14) crosses below -20 (overbought rejection) AND volume > 1.8x 20-period average AND 4h EMA20 < EMA50 (downtrend)
# Exit when Williams %R crosses above -20 (for longs) or below -80 (for shorts) OR trend flips
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-35 trades/year per symbol.
# Williams %R identifies exhaustion points, volume confirms participation, 4h EMA cross filters for higher timeframe momentum alignment.

name = "12h_WilliamsR_Volume_4hEMA_Cross"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low + 1e-10) * -100
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 4h data for EMA cross trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA20 and EMA50 on 4h data
    close_4h = df_4h['close'].values
    ema_20 = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Uptrend when EMA20 > EMA50, downtrend when EMA20 < EMA50
    uptrend_4h = ema_20 > ema_50
    downtrend_4h = ema_20 < ema_50
    
    # Align 4h trend to 12h timeframe
    uptrend_4h_aligned = align_htf_to_ltf(prices, df_4h, uptrend_4h.astype(float))
    downtrend_4h_aligned = align_htf_to_ltf(prices, df_4h, downtrend_4h.astype(float))
    
    # Volume confirmation: volume > 1.8x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(uptrend_4h_aligned[i]) or 
            np.isnan(downtrend_4h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        williams_r_val = williams_r_aligned[i]
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (from below) AND volume spike AND 4h uptrend
            if (williams_r_val > -80 and 
                williams_r_aligned[i-1] <= -80 and 
                volume_filter[i] and 
                uptrend_4h_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 (from above) AND volume spike AND 4h downtrend
            elif (williams_r_val < -20 and 
                  williams_r_aligned[i-1] >= -20 and 
                  volume_filter[i] and 
                  downtrend_4h_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 OR 4h trend flips to downtrend
            if (williams_r_val > -20 and williams_r_aligned[i-1] <= -20) or \
               downtrend_4h_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 OR 4h trend flips to uptrend
            if (williams_r_val < -80 and williams_r_aligned[i-1] >= -80) or \
               uptrend_4h_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals