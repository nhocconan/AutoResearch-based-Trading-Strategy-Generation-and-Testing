#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R Extreme Reversal with 1d EMA34 trend filter and volume confirmation (1.5x)
# Long when Williams %R(14) crosses above -80 (oversold reversal) AND price > 1d EMA34 AND volume > 1.5x 20-period average
# Short when Williams %R(14) crosses below -20 (overbought reversal) AND price < 1d EMA34 AND volume > 1.5x 20-period average
# Exit when Williams %R returns to -50 (mean reversion) OR 1d EMA34 filter reverses
# Williams %R captures momentum extremes effective in both bull and bear markets
# 1d EMA34 provides higher timeframe trend filter
# Volume confirmation reduces false signals
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Timeframe: 12h (primary), HTF: 1d

name = "12h_WilliamsR_Extreme_1dEMA34_VolumeSpike_1.5x"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF EMA to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 12h timeframe
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, -50)
    
    # Williams %R signals: crossover above -80 (long) or below -20 (short)
    williams_r_long_signal = (williams_r > -80) & (np.roll(williams_r, 1) <= -80)
    williams_r_short_signal = (williams_r < -20) & (np.roll(williams_r, 1) >= -20)
    williams_r_exit_signal = np.abs(williams_r + 50) < 2.5  # Exit near -50
    
    # Volume confirmation on 12h (threshold: 1.5x for optimal frequency)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND price > EMA34 AND volume spike
            if (williams_r_long_signal[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < EMA34 AND volume spike
            elif (williams_r_short_signal[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 OR price < EMA34 (trend weakening)
            if williams_r_exit_signal[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 OR price > EMA34 (trend weakening)
            if williams_r_exit_signal[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals