#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# Long when: 6h Williams %R < -80 (oversold) AND 1d close > 1d EMA34 (bullish trend) AND volume > 1.5x 20-period MA
# Short when: 6h Williams %R > -20 (overbought) AND 1d close < 1d EMA34 (bearish trend) AND volume > 1.5x 20-period MA
# Exit when: Williams %R returns to -50 (mean reversion midpoint) OR opposite extreme reached
# Uses Williams %R for mean reversion timing, 1d EMA for trend filter, volume for conviction
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WilliamsR_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R on 6h (14-period)
    if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when high == low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Bullish bias: close > EMA34, Bearish bias: close < EMA34
    bullish_bias = df_1d['close'].values > ema_34_1d
    bearish_bias = df_1d['close'].values < ema_34_1d
    
    # Align 1d EMA bias to 6h timeframe
    bullish_bias_aligned = align_htf_to_ltf(prices, df_1d, bullish_bias.astype(float))
    bearish_bias_aligned = align_htf_to_ltf(prices, df_1d, bearish_bias.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(bullish_bias_aligned[i]) or 
            np.isnan(bearish_bias_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold + bullish bias + volume filter
            if (williams_r[i] < -80 and 
                bullish_bias_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought + bearish bias + volume filter
            elif (williams_r[i] > -20 and 
                  bearish_bias_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to midpoint (-50) OR short entry signal
            if (williams_r[i] >= -50 or 
                (williams_r[i] > -20 and bearish_bias_aligned[i] == 1.0 and volume_filter[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to midpoint (-50) OR long entry signal
            if (williams_r[i] <= -50 or 
                (williams_r[i] < -80 and bullish_bias_aligned[i] == 1.0 and volume_filter[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals