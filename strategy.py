#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 12h EMA50 trend filter and volume confirmation (1.8x)
# Long when Williams %R(14) crosses above -80 (oversold reversal) AND price > 12h EMA50 AND volume > 1.8x 20-period average
# Short when Williams %R(14) crosses below -20 (overbought reversal) AND price < 12h EMA50 AND volume > 1.8x 20-period average
# Exit when Williams %R returns to -50 (mean reversion) OR 12h EMA50 filter reverses
# Williams %R captures exhaustion points in both bull and bear markets, EMA50 filters counter-trend noise
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 6h (primary), HTF: 12h

name = "6h_WilliamsR_Extreme_Reversal_12hEMA50_VolumeSpike_1.8x"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation on 6h (threshold: 1.8x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.8 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    # Williams %R(14) on 6h
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Avoid division by zero
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, -50)
    
    # Williams %R cross signals
    williams_r_cross_up = (williams_r > -80) & (np.roll(williams_r, 1) <= -80)  # Cross above -80
    williams_r_cross_down = (williams_r < -20) & (np.roll(williams_r, 1) >= -20)  # Cross below -20
    williams_r_mean_revert = (williams_r > -50) & (np.roll(williams_r, 1) <= -50)  # Cross above -50 for exit long
    williams_r_mean_revert_short = (williams_r < -50) & (np.roll(williams_r, 1) >= -50)  # Cross below -50 for exit short
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND price > EMA50 AND volume spike
            if (williams_r_cross_up[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < EMA50 AND volume spike
            elif (williams_r_cross_down[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R mean reversion (cross above -50) OR price < EMA50 (trend weakening)
            if williams_r_mean_revert[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R mean reversion (cross below -50) OR price > EMA50 (trend weakening)
            if williams_r_mean_revert_short[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals