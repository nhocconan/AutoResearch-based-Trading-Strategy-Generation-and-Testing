#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme reversal with 1d EMA200 trend filter and volume spike confirmation
# Long when Williams %R < -80 (oversold) AND price > EMA200(1d) AND volume > 2.0x 20-period average
# Short when Williams %R > -20 (overbought) AND price < EMA200(1d) AND volume > 2.0x 20-period average
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR price crosses EMA200(1d) in opposite direction
# Williams %R identifies extreme price reversals, effective in both bull and bear markets
# 1d EMA200 provides strong higher timeframe trend filter to avoid counter-trend trades
# Volume spike confirms institutional participation during reversal
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) for 4h timeframe
# Discrete sizing (0.30) to balance profit potential and fee drag

name = "4h_WilliamsR_EXTREME_1dEMA200_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA200 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 4h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Williams %R (14 period)
    if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    # Volume confirmation: volume > 2.0x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND price > EMA200(1d) AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_200_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND price < EMA200(1d) AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR price < EMA200(1d) (trend flip)
            if (williams_r[i] > -50 or 
                close[i] < ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR price > EMA200(1d) (trend flip)
            if (williams_r[i] < -50 or 
                close[i] > ema_200_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals