# 6h_Camarilla_R3_S3_Breakout_1dTrend_Force
# Strategy: Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Rationale: Camarilla levels act as intraday support/resistance. Breakouts from R3/S3
# with trend alignment and volume capture momentum moves in both bull and bear markets.
# 6h timeframe reduces noise, 1d trend filter ensures directional bias, volume confirms strength.
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Force"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Standard Camarilla: based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    S4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (no extra delay - levels based on closed 1d bar)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: volume > 1.5x 20-period moving average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # LONG: break above R3 with trend up and volume
        if close[i] > R3_aligned[i] and close[i] > ema_34_1d_aligned[i] and vol_spike:
            signals[i] = 0.25
        # SHORT: break below S3 with trend down and volume
        elif close[i] < S3_aligned[i] and close[i] < ema_34_1d_aligned[i] and vol_spike:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals