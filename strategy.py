#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d EMA trend + volume spike
# Williams %R(14): overbought > -20, oversold < -80
# Long: %R crosses above -80 from below AND price > 1d EMA(34) AND volume > 2x 20-period MA
# Short: %R crosses below -20 from above AND price < 1d EMA(34) AND volume > 2x 20-period MA
# Exit: %R crosses opposite extreme (-20 for long, -80 for short) OR volume drops below average
# Uses mean reversion in extreme zones with trend filter and volume confirmation
# Timeframe: 6h, HTF: 1d for EMA. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_1dEMA_VolumeSpike"
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
    
    # Williams %R(14) on 6h
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when high == low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short)
    williams_long_signal = (williams_r > -80) & (np.roll(williams_r, 1) <= -80)
    williams_short_signal = (williams_r < -20) & (np.roll(williams_r, 1) >= -20)
    # Handle first bar
    williams_long_signal[0] = False
    williams_short_signal[0] = False
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
        volume_average = volume <= (1.5 * vol_ma_20)  # for exit
    else:
        volume_spike = np.zeros(n, dtype=bool)
        volume_average = np.ones(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Trend filter: price above/below 1d EMA(34)
    price_above_ema = close > ema_34_1d_aligned
    price_below_ema = close < ema_34_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(volume_average[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 + price > EMA + volume spike
            if (williams_long_signal[i] and 
                price_above_ema[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 + price < EMA + volume spike
            elif (williams_short_signal[i] and 
                  price_below_ema[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 OR volume drops to average
            if (williams_r[i] >= -20 or volume_average[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 OR volume drops to average
            if (williams_r[i] <= -80 or volume_average[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals